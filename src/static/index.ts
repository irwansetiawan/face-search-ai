import * as async from 'async';
import { saveAs } from 'file-saver';
import JSZip from 'jszip';

/* ===================================================================
   Face Search — browser layer.

   The network contract is load-bearing and unchanged from the version
   this UI replaced:
     - exactly ONE /probe per scan when searching by uploaded photo
     - ZERO /probe calls when searching by saved person (the embedding
       already exists server-side — that is the whole point of saving)
     - one /search per photo, at concurrency 2
     - every failure counted, so a scan that broke can never be mistaken
       for a scan that found nobody
   =================================================================== */

type RelativeBox = { top: number; left: number; width: number; height: number };
type SourceRef = { probe: number[] } | { personId: string };

interface ScoredFace {
    box: RelativeBox;
    cosine: number;
    detScore: number;
    matched: boolean;
    bestReference: number;
}

interface SearchBody {
    threshold: number;
    matched: boolean;
    faces: ScoredFace[];
}

interface PersonReference { id: string; thumb: string; detScore: number; addedAt: string; }
interface PersonSummary { id: string; name: string; createdAt: string; references: PersonReference[]; }

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/* --------------------------------------------------------- elements */

const sourceDrop     = $<HTMLDivElement>('sourceDrop');
const sourceInput    = $<HTMLInputElement>('sourceInput');
const sourceFigure   = $<HTMLElement>('sourceFigure');
const sourceImg      = $<HTMLImageElement>('sourceImg');
const sourceHint     = $<HTMLParagraphElement>('sourceHint');
const clearSourceBtn = $<HTMLButtonElement>('clearSourceBtn');

const faceprintEl     = $<HTMLDivElement>('faceprint');
const faceprintCanvas = $<HTMLCanvasElement>('faceprintCanvas');
const faceprintNote   = $<HTMLParagraphElement>('faceprintNote');
const saveToggleBtn   = $<HTMLButtonElement>('saveToggleBtn');
const saveRow         = $<HTMLDivElement>('saveRow');
const newPersonName   = $<HTMLInputElement>('newPersonName');
const savePersonBtn   = $<HTMLButtonElement>('savePersonBtn');

const peopleRail  = $<HTMLDivElement>('peopleRail');
const peopleCount = $<HTMLSpanElement>('peopleCount');

const targetDrop        = $<HTMLDivElement>('targetDrop');
const targetFilesInput  = $<HTMLInputElement>('targetFilesInput');
const targetFolderInput = $<HTMLInputElement>('targetFolderInput');
const targetHint        = $<HTMLParagraphElement>('targetHint');
const targetStrip       = $<HTMLDivElement>('targetStrip');
const targetCount       = $<HTMLSpanElement>('targetCount');
const pickFilesBtn      = $<HTMLButtonElement>('pickFilesBtn');
const pickFolderBtn     = $<HTMLButtonElement>('pickFolderBtn');

const goBtn      = $<HTMLButtonElement>('goBtn');
const meterFill  = $<HTMLDivElement>('meterFill');
const gScanned   = $<HTMLSpanElement>('gScanned');
const gFound     = $<HTMLSpanElement>('gFound');
const gFailed    = $<HTMLSpanElement>('gFailed');
const gaugeFailed = $<HTMLDivElement>('gaugeFailed');

const sheet          = $<HTMLDivElement>('sheet');
const sheetEmpty     = $<HTMLDivElement>('sheetEmpty');
const sheetNote      = $<HTMLSpanElement>('sheetNote');
const saveMatchesBtn = $<HTMLButtonElement>('saveMatchesBtn');

const thresholdChip  = $<HTMLSpanElement>('thresholdChip');
const thresholdValue = $<HTMLElement>('thresholdValue');

const lightbox       = $<HTMLDivElement>('lightbox');
const lightboxImg    = $<HTMLImageElement>('lightboxImg');
const lightboxCanvas = $<HTMLCanvasElement>('lightboxCanvas');
const lightboxName   = $<HTMLSpanElement>('lightboxName');
const lightboxScore  = $<HTMLSpanElement>('lightboxScore');
const lightboxClose  = $<HTMLButtonElement>('lightboxClose');

const notices = $<HTMLDivElement>('notices');
const bench   = $<HTMLFormElement>('bench');

/* ------------------------------------------------------------ state */

let sourceFile: File | null = null;
let selectedPersonId: string | null = null;
let targetFiles: File[] = [];
let scanning = false;

const objectUrls: string[] = [];
function keepUrl(file: Blob): string {
    const url = URL.createObjectURL(file);
    objectUrls.push(url);
    return url;
}
function releaseUrls() {
    while (objectUrls.length) URL.revokeObjectURL(objectUrls.pop()!);
}

const counters = { total: 0, responsesReceived: 0, filesMatched: 0, failed: 0 };
let matchedFiles: File[] = [];

/* ---------------------------------------------------------- notices */

function notify(message: string, bad = false) {
    const el = document.createElement('div');
    el.className = bad ? 'notice notice--bad' : 'notice';
    el.textContent = message;
    notices.appendChild(el);
    setTimeout(() => el.remove(), 5200);
}

/* ------------------------------------------------------- faceprint */
/* The signature element. Every spoke is one of the 512 dimensions the
   model actually produced for this face — length from magnitude, colour
   from sign. It is the face as the machine sees it, not an ornament. */

function drawFaceprint(embedding: number[]) {
    const ctx = faceprintCanvas.getContext('2d');
    if (!ctx) return;

    const size = faceprintCanvas.width;
    const mid = size / 2;
    const inner = size * 0.20;
    const outer = size * 0.46;
    const peak = Math.max(...embedding.map(Math.abs)) || 1;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    ctx.clearRect(0, 0, size, size);
    ctx.lineWidth = 1;

    const spoke = (i: number) => {
        const v = embedding[i];
        const angle = (i / embedding.length) * Math.PI * 2 - Math.PI / 2;
        const len = inner + (Math.abs(v) / peak) * (outer - inner);
        ctx.strokeStyle = v >= 0 ? 'rgba(21,24,27,.72)' : 'rgba(53,86,110,.72)';
        ctx.beginPath();
        ctx.moveTo(mid + Math.cos(angle) * inner, mid + Math.sin(angle) * inner);
        ctx.lineTo(mid + Math.cos(angle) * len, mid + Math.sin(angle) * len);
        ctx.stroke();
    };

    const ring = () => {
        ctx.strokeStyle = 'rgba(110,117,112,.5)';
        ctx.beginPath();
        ctx.arc(mid, mid, inner - 3, 0, Math.PI * 2);
        ctx.stroke();
    };

    if (reduce) {
        for (let i = 0; i < embedding.length; i++) spoke(i);
        ring();
        return;
    }

    let drawn = 0;
    const step = () => {
        const target = Math.min(embedding.length, drawn + 22);
        for (; drawn < target; drawn++) spoke(drawn);
        if (drawn < embedding.length) requestAnimationFrame(step);
        else ring();
    };
    requestAnimationFrame(step);
}

/* ------------------------------------------------- source selection */

function setSourceFile(file: File | null) {
    sourceFile = file;
    if (file) {
        // A photo and a saved person are alternatives, not a combination.
        selectedPersonId = null;
        markPickedPerson();
        sourceImg.src = keepUrl(file);
        sourceFigure.hidden = false;
        sourceHint.hidden = true;
        sourceDrop.classList.add('has-image');
        clearSourceBtn.hidden = false;
    } else {
        sourceImg.removeAttribute('src');
        sourceFigure.hidden = true;
        sourceHint.hidden = false;
        sourceDrop.classList.remove('has-image');
        clearSourceBtn.hidden = true;
    }
    clearSourceOverlay();
    faceprintEl.classList.remove('is-on');
    saveRow.hidden = true;
}

function clearSourceOverlay() {
    sourceFigure.querySelector('canvas')?.remove();
}

function drawSourceBox(box: RelativeBox) {
    clearSourceOverlay();
    const canvas = document.createElement('canvas');
    const w = sourceImg.clientWidth, h = sourceImg.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    if (ctx) {
        ctx.scale(dpr, dpr);
        ctx.strokeStyle = '#C8402C';
        ctx.lineWidth = 2;
        ctx.strokeRect(box.left * w, box.top * h, box.width * w, box.height * h);
    }
    sourceFigure.appendChild(canvas);
}

sourceDrop.addEventListener('click', () => sourceInput.click());
sourceDrop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sourceInput.click(); }
});
sourceInput.addEventListener('change', () => {
    const f = sourceInput.files?.[0];
    if (f) setSourceFile(f);
});
clearSourceBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    sourceInput.value = '';
    setSourceFile(null);
});

/* -------------------------------------------------- target selection */

const IMAGE_RE = /\.(jpe?g|png)$/i;

const STRIP_MAX = 12;

function setTargetFiles(files: File[]) {
    targetFiles = files.filter(f => IMAGE_RE.test(f.name));
    targetStrip.innerHTML = '';

    if (targetFiles.length === 0) {
        targetCount.textContent = '';
        targetStrip.hidden = true;
        targetHint.innerHTML = '<b>Drop photos, or a whole folder</b>JPEG and PNG';
        return;
    }

    const n = targetFiles.length;
    targetCount.textContent = `${n} ready`;
    targetHint.innerHTML =
        `<b>${n} photo${n === 1 ? '' : 's'} ready</b>Drop more to replace this set`;

    // Show what was actually chosen. Only the first few get object URLs —
    // a 500-photo folder must not mint 500 of them just for a preview.
    for (const file of targetFiles.slice(0, STRIP_MAX)) {
        const img = document.createElement('img');
        img.src = keepUrl(file);
        img.alt = '';
        targetStrip.appendChild(img);
    }
    if (n > STRIP_MAX) {
        const more = document.createElement('span');
        more.className = 'strip-more';
        more.textContent = `+${n - STRIP_MAX}`;
        targetStrip.appendChild(more);
    }
    targetStrip.hidden = false;
}

targetDrop.addEventListener('click', () => targetFilesInput.click());
targetDrop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); targetFilesInput.click(); }
});
pickFilesBtn.addEventListener('click', (e) => { e.stopPropagation(); targetFilesInput.click(); });
pickFolderBtn.addEventListener('click', (e) => { e.stopPropagation(); targetFolderInput.click(); });
targetFilesInput.addEventListener('change', () => setTargetFiles(Array.from(targetFilesInput.files ?? [])));
targetFolderInput.addEventListener('change', () => setTargetFiles(Array.from(targetFolderInput.files ?? [])));

/* ------------------------------------------------------- drag & drop */

function wireDrop(zone: HTMLElement, onFiles: (files: File[]) => void) {
    ['dragenter', 'dragover'].forEach(ev =>
        zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('is-over'); }));
    ['dragleave', 'drop'].forEach(ev =>
        zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('is-over'); }));
    zone.addEventListener('drop', (e) => {
        const dt = (e as DragEvent).dataTransfer;
        if (!dt) return;
        onFiles(Array.from(dt.files));
    });
}

wireDrop(sourceDrop, (files) => { if (files[0]) setSourceFile(files[0]); });
wireDrop(targetDrop, (files) => setTargetFiles(files));

/* ------------------------------------------------------ saved people */

function markPickedPerson() {
    let pickedName: string | null = null;
    peopleRail.querySelectorAll('.person').forEach(el => {
        const on = (el as HTMLElement).dataset.id === selectedPersonId;
        el.classList.toggle('is-picked', on);
        if (on) pickedName = (el as HTMLElement).dataset.name ?? null;
    });

    // The two inputs are alternatives, so the source plate has to say which
    // one is live — otherwise it reads "drop a photo" while a saved person
    // is what the scan will actually use.
    if (pickedName && !sourceFile) {
        sourceHint.innerHTML =
            `<b>Searching for ${escapeHtml(pickedName)}</b>Drop a photo here to search for someone else instead`;
    } else if (!sourceFile) {
        sourceHint.innerHTML = '<b>Drop a photo of one person</b>or click to choose';
    }
}

function escapeHtml(s: string): string {
    const d = document.createElement('div');
    d.innerText = s;
    return d.innerHTML;
}

async function loadPeople(): Promise<void> {
    let people: PersonSummary[];
    try {
        const res = await fetch('/people');
        if (!res.ok) throw new Error('bad status ' + res.status);
        people = await res.json();
    } catch (e) {
        console.warn('Could not load saved people', e);
        peopleRail.innerHTML = '<p class="people-empty">Saved people could not be loaded.</p>';
        // A failed reload must not leave a selection alive: the guard on
        // submit would pass, /search would get a personId the server can't
        // resolve, and a whole scan would silently report "found 0".
        selectedPersonId = null;
        return;
    }

    // Same reasoning on success: a person picked earlier may have been
    // deleted since (another tab, or data/ wiped).
    if (selectedPersonId && !people.some(p => p.id === selectedPersonId)) {
        selectedPersonId = null;
    }

    peopleCount.textContent = people.length ? `${people.length}` : '';
    peopleRail.innerHTML = '';

    if (people.length === 0) {
        peopleRail.innerHTML =
            '<p class="people-empty">None yet. Find a face, then save it to reuse later.</p>';
        return;
    }

    for (const person of people) {
        const el = document.createElement('div');
        el.className = 'person';
        el.dataset.id = person.id;
        el.dataset.name = person.name;
        el.innerHTML =
            `<button type="button" class="person-btn">` +
            `<img src="/people-files/${person.references[0].thumb}" alt="${escapeHtml(person.name)}"></button>` +
            `<p class="person-name">${escapeHtml(person.name)}</p>` +
            `<p class="person-meta">${person.references.length} photo${person.references.length === 1 ? '' : 's'}</p>` +
            `<div class="person-tools">` +
            `<button type="button" class="linkish add-ref">add</button>` +
            `<button type="button" class="linkish manage-refs">edit</button>` +
            `</div>` +
            `<input type="file" accept="image/*" class="add-ref-input visually-hidden">` +
            `<div class="refs" hidden></div>`;

        el.querySelector('.person-btn')!.addEventListener('click', () => {
            selectedPersonId = person.id;
            sourceInput.value = '';
            setSourceFile(null);          // a person replaces an uploaded photo
            selectedPersonId = person.id; // setSourceFile(null) doesn't clear this
            markPickedPerson();
        });

        // Adding a reference photo is how max-over-references becomes
        // reachable at all — a person with several angles matches far more
        // reliably than one with a single frame.
        const addInput = el.querySelector('.add-ref-input') as HTMLInputElement;
        const addBtn = el.querySelector('.add-ref') as HTMLButtonElement;
        addBtn.addEventListener('click', () => addInput.click());
        addInput.addEventListener('change', async () => {
            const file = addInput.files?.[0];
            if (!file) return;
            addBtn.disabled = true;
            try {
                const fd = new FormData();
                fd.append('photo', file);
                const res = await fetch(`/people/${person.id}/references`, { method: 'POST', body: fd });
                if (res.status === 422) { notify('No face found in that photo.', true); return; }
                if (res.status === 404) { notify('That person no longer exists.', true); await loadPeople(); return; }
                if (!res.ok) { notify('Could not add that photo.', true); return; }
                notify(`Added a photo to ${person.name}.`);
                await loadPeople();
                markPickedPerson();
            } catch {
                notify('Could not reach the server.', true);
            } finally {
                addBtn.disabled = false;
                addInput.value = '';
            }
        });

        const manageBtn = el.querySelector('.manage-refs') as HTMLButtonElement;
        const refsPanel = el.querySelector('.refs') as HTMLDivElement;
        manageBtn.addEventListener('click', () => {
            if (!refsPanel.hidden) { refsPanel.hidden = true; return; }
            refsPanel.innerHTML = '';
            for (const ref of person.references) {
                const row = document.createElement('div');
                row.className = 'ref-row';
                row.innerHTML =
                    `<img src="/people-files/${ref.thumb}" alt="">` +
                    `<button type="button" class="linkish linkish--danger">remove</button>`;
                row.querySelector('button')!.addEventListener('click', async () => {
                    if (!confirm('Remove this reference photo?')) return;
                    try {
                        const res = await fetch(`/people/${person.id}/references/${ref.id}`, { method: 'DELETE' });
                        if (res.status === 409) {
                            notify('That is the only photo left. Delete the person instead.', true); return;
                        }
                        if (res.status === 404) { notify('That photo no longer exists.', true); await loadPeople(); return; }
                        if (!res.ok) { notify('Could not remove that photo.', true); return; }
                        await loadPeople();
                        markPickedPerson();
                    } catch {
                        notify('Could not reach the server.', true);
                    }
                });
                refsPanel.appendChild(row);
            }
            const del = document.createElement('button');
            del.type = 'button';
            del.className = 'linkish linkish--danger';
            del.textContent = `Delete ${person.name}`;
            del.style.marginTop = '6px';
            del.addEventListener('click', async () => {
                if (!confirm(`Delete ${person.name}? This removes their photos from this machine.`)) return;
                try {
                    const res = await fetch(`/people/${person.id}`, { method: 'DELETE' });
                    if (!res.ok) { notify('Could not delete that person.', true); return; }
                } catch {
                    notify('Could not reach the server.', true); return;
                }
                if (selectedPersonId === person.id) selectedPersonId = null;
                notify(`Deleted ${person.name}.`);
                await loadPeople();
            });
            refsPanel.appendChild(del);
            refsPanel.hidden = false;
        });

        peopleRail.appendChild(el);
    }
    markPickedPerson();
}

/* ------------------------------------------------------ saving a face */

saveToggleBtn.addEventListener('click', () => {
    saveRow.hidden = !saveRow.hidden;
    if (!saveRow.hidden) newPersonName.focus();
});

savePersonBtn.addEventListener('click', async () => {
    const name = newPersonName.value.trim();
    if (!name) { notify('Give this person a name first.', true); return; }
    if (!sourceFile) { notify('Choose a photo first.', true); return; }
    savePersonBtn.disabled = true;
    try {
        const fd = new FormData();
        fd.append('photo', sourceFile);
        fd.append('name', name);
        const res = await fetch('/people', { method: 'POST', body: fd });
        if (res.status === 422) { notify('No face found in that photo.', true); return; }
        if (!res.ok) { notify('Could not save that person.', true); return; }
        notify(`Saved ${name}. They will be here next time.`);
        newPersonName.value = '';
        saveRow.hidden = true;
        await loadPeople();
    } catch {
        notify('Could not reach the server.', true);
    } finally {
        savePersonBtn.disabled = false;
    }
});

/* --------------------------------------------------- the contact sheet */

interface Cell { file: File; url: string; body?: SearchBody; }
const cells: Cell[] = [];

function ellipsePerimeter(a: number, b: number) {
    return Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
}

/** Place the chinagraph mark over the matched face, accounting for the
 *  letterboxing that object-fit: contain introduces. */
function markFace(shot: HTMLElement, img: HTMLImageElement, box: RelativeBox) {
    const cw = shot.clientWidth, ch = shot.clientHeight;
    const nw = img.naturalWidth, nh = img.naturalHeight;
    if (!cw || !ch || !nw || !nh) return;

    const scale = Math.min(cw / nw, ch / nh);
    const rw = nw * scale, rh = nh * scale;
    const ox = (cw - rw) / 2, oy = (ch - rh) / 2;

    const x = ox + box.left * rw, y = oy + box.top * rh;
    const w = box.width * rw, h = box.height * rh;
    const rx = Math.max(w / 2 * 1.28, 7), ry = Math.max(h / 2 * 1.22, 7);

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'mark');
    svg.setAttribute('width', String(cw));
    svg.setAttribute('height', String(ch));
    svg.style.left = '0'; svg.style.top = '0';

    const ell = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
    ell.setAttribute('cx', String(x + w / 2));
    ell.setAttribute('cy', String(y + h / 2));
    ell.setAttribute('rx', String(rx));
    ell.setAttribute('ry', String(ry));
    ell.setAttribute('transform', `rotate(-6 ${x + w / 2} ${y + h / 2})`);
    ell.style.setProperty('--dash', String(Math.ceil(ellipsePerimeter(rx, ry))));

    svg.appendChild(ell);
    shot.appendChild(svg);
}

function addCell(file: File, body: SearchBody | undefined, failed: boolean) {
    sheetEmpty.hidden = true;
    const url = keepUrl(file);
    const cell: Cell = { file, url, body };
    cells.push(cell);

    const best = body?.faces?.find(f => f.matched)
        ?? body?.faces?.slice().sort((a, b) => b.cosine - a.cosine)[0];
    const hit = !!body?.matched;

    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'frame ' + (hit ? 'frame--hit' : 'frame--miss');
    el.innerHTML =
        `<div class="frame-shot"><img alt="${escapeHtml(file.name)}"></div>` +
        `<div class="frame-foot">` +
        `<span class="frame-name">${escapeHtml(file.name)}</span>` +
        `<span class="frame-score${hit ? '' : ' frame-score--miss'}">` +
        (failed ? 'failed' : best ? best.cosine.toFixed(3) : 'no face') +
        `</span></div>`;

    const shot = el.querySelector('.frame-shot') as HTMLElement;
    const img = el.querySelector('img') as HTMLImageElement;
    img.addEventListener('load', () => {
        if (hit && best) markFace(shot, img, best.box);
    });
    img.src = url;

    el.addEventListener('click', () => openLightbox(cell));
    sheet.appendChild(el);
}

/* ------------------------------------------------------------ lightbox */

function openLightbox(cell: Cell) {
    lightboxImg.src = cell.url;
    lightboxName.textContent = cell.file.name;
    const best = cell.body?.faces?.find(f => f.matched);
    lightboxScore.textContent = best
        ? `${best.cosine.toFixed(3)} cosine`
        : cell.body ? `${cell.body.faces.length} face${cell.body.faces.length === 1 ? '' : 's'}, no match` : '';
    lightbox.classList.add('is-on');

    const paint = () => {
        const w = lightboxImg.clientWidth, h = lightboxImg.clientHeight;
        if (!w || !h) return;
        const dpr = window.devicePixelRatio || 1;
        // Backing store only — CSS (inset:0; width/height:100%) stretches the
        // canvas over the figure. Setting explicit pixel sizes here fights
        // that and leaves the boxes overhanging the photo after a resize.
        // Every coordinate below is a fraction of w/h, so a proportional
        // stretch keeps the boxes on their faces at any size.
        lightboxCanvas.width = w * dpr;
        lightboxCanvas.height = h * dpr;
        const ctx = lightboxCanvas.getContext('2d');
        if (!ctx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        for (const face of cell.body?.faces ?? []) {
            ctx.strokeStyle = face.matched ? '#C8402C' : 'rgba(110,117,112,.85)';
            ctx.lineWidth = face.matched ? 2.5 : 1.5;
            ctx.strokeRect(face.box.left * w, face.box.top * h, face.box.width * w, face.box.height * h);
            if (face.matched) {
                // Photos are unpredictable behind a label; a dark halo keeps
                // the score readable over a bright window or a white shirt.
                ctx.font = '500 13px "DM Mono", monospace';
                ctx.shadowColor = 'rgba(0,0,0,.85)';
                ctx.shadowBlur = 6;
                ctx.fillStyle = '#FFFFFF';
                ctx.fillText(face.cosine.toFixed(3), face.box.left * w, Math.max(14, face.box.top * h - 7));
                ctx.shadowBlur = 0;
            }
        }
    };

    if (lightboxImg.complete) paint(); else lightboxImg.addEventListener('load', paint, { once: true });
}

function closeLightbox() { lightbox.classList.remove('is-on'); }
lightboxClose.addEventListener('click', closeLightbox);
lightbox.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });

/* -------------------------------------------------------- the scan */

function resetScan(total: number) {
    counters.total = total;
    counters.responsesReceived = 0;
    counters.filesMatched = 0;
    counters.failed = 0;
    matchedFiles = [];
    cells.length = 0;
    sheet.innerHTML = '';
    sheetEmpty.hidden = false;
    saveMatchesBtn.hidden = true;
    meterFill.classList.remove('is-done');
    render();
}

function render() {
    const pct = counters.total > 0
        ? Math.ceil((counters.responsesReceived / counters.total) * 100) : 0;
    meterFill.style.width = pct + '%';
    gScanned.textContent = `${counters.responsesReceived}`;
    gFound.textContent = `${counters.filesMatched}`;
    gFailed.textContent = `${counters.failed}`;

    // A scan that failed must never look like a scan that found nobody.
    gaugeFailed.classList.toggle('is-on', counters.failed > 0);
    // Chinagraph only once there is something to mark.
    gFound.parentElement!.classList.toggle('is-hot', counters.filesMatched > 0);

    sheetNote.textContent = counters.total
        ? `${counters.responsesReceived} of ${counters.total} scanned`
        : '';
}

async function runSearch(source: SourceRef, file: File): Promise<void> {
    let body: SearchBody | undefined;
    let failed = false;
    try {
        const fd = new FormData();
        fd.append('target', file);
        if ('personId' in source) fd.append('personId', source.personId);
        else fd.append('probe', JSON.stringify([source.probe]));

        const res = await fetch('/search', { method: 'POST', body: fd });
        const parsed = await res.json();
        if (!res.ok) {
            // Counted, not just logged: a run where every request is
            // rejected must not report a clean "found nobody".
            counters.failed += 1;
            failed = true;
            console.warn(file.name, parsed.error);
        } else {
            body = parsed as SearchBody;
            if (thresholdChip.hidden && typeof body.threshold === 'number') {
                thresholdValue.textContent = body.threshold.toFixed(2);
                thresholdChip.hidden = false;
            }
            if (body.matched) {
                counters.filesMatched += 1;
                matchedFiles.push(file);
            }
        }
    } catch (error) {
        // A network or parse failure for one photo must not abort the run.
        counters.failed += 1;
        failed = true;
        console.warn(file.name, 'search request failed', error);
    } finally {
        counters.responsesReceived += 1;
        addCell(file, body, failed);
        render();
    }
}

async function startScan() {
    if (scanning) { notify('A scan is already running.'); return; }

    const usingPerson = !!selectedPersonId;
    if (!usingPerson && !sourceFile) { notify('Choose a photo, or pick a saved person.', true); return; }
    if (targetFiles.length === 0) { notify('Choose the photos to search through.', true); return; }

    let source: SourceRef;

    if (usingPerson) {
        // ZERO /probe calls here — the saved person's embedding already
        // exists server-side.
        source = { personId: selectedPersonId! };
    } else {
        // Exactly ONE /probe for the entire run, outside the per-photo loop.
        try {
            const fd = new FormData();
            fd.append('source', sourceFile!);
            const res = await fetch('/probe', { method: 'POST', body: fd });
            if (res.status === 422) { notify('No face found in that photo. Try a clearer one.', true); return; }
            if (!res.ok) { notify('Could not read that photo.', true); return; }
            const json = await res.json();
            drawSourceBox(json.face.box);
            drawFaceprint(json.embedding);
            faceprintNote.textContent = `${json.embedding.length} dimensions · detected at ${json.face.score.toFixed(2)}`;
            faceprintEl.classList.add('is-on');
            source = { probe: json.embedding };
        } catch {
            notify('Could not reach the server.', true); return;
        }
    }

    scanning = true;
    goBtn.disabled = true;
    goBtn.textContent = 'Scanning…';
    resetScan(targetFiles.length);

    const files = targetFiles.slice();
    async.eachLimit(files, 2, (file: File, callback) => {
        runSearch(source, file).then(() => callback()).catch((err) => callback(err));
    }, () => {
        scanning = false;
        goBtn.disabled = false;
        goBtn.textContent = 'Find matches';
        meterFill.classList.add('is-done');
        saveMatchesBtn.hidden = matchedFiles.length === 0;
        saveMatchesBtn.textContent =
            `Save ${matchedFiles.length} photo${matchedFiles.length === 1 ? '' : 's'} as a zip`;

        if (counters.failed > 0) {
            notify(`${counters.failed} photo${counters.failed === 1 ? '' : 's'} could not be scanned — results are incomplete.`, true);
        } else if (matchedFiles.length === 0) {
            notify('No matches in this set.');
        } else {
            notify(`Found ${matchedFiles.length} photo${matchedFiles.length === 1 ? '' : 's'}.`);
        }

        // One photo in, one photo out: open it rather than making the
        // reader click a sheet of one.
        if (cells.length === 1) openLightbox(cells[0]);
    });
}

goBtn.addEventListener('click', startScan);
bench.addEventListener('submit', (e) => { e.preventDefault(); startScan(); });

/* Saving matches is an action the reader takes, not a download that
   happens to them when a scan ends. */
saveMatchesBtn.addEventListener('click', async () => {
    if (matchedFiles.length === 0) return;
    saveMatchesBtn.disabled = true;
    try {
        const zip = new JSZip();
        for (const file of matchedFiles) zip.file(file.name, file);
        const blob = await zip.generateAsync({ type: 'blob' });
        saveAs(blob, `face-search-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.zip`);
    } finally {
        saveMatchesBtn.disabled = false;
    }
});

window.addEventListener('beforeunload', releaseUrls);

loadPeople();
render();
