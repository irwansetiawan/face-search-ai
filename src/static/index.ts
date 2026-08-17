import * as async from 'async';
import { saveAs } from 'file-saver';
import JSZip from 'jszip';

// source image
const sourceInput = document.getElementById('source') as HTMLInputElement;

// source preview
const sourceImg = document.getElementById('sourceImg') as HTMLImageElement;

// source mode: upload a photo vs. pick a saved person
const radioSourceUpload = document.getElementById('sourceTypeUpload') as HTMLInputElement;
const radioSourcePerson = document.getElementById('sourceTypePerson') as HTMLInputElement;
const peoplePicker = document.getElementById('peoplePicker') as HTMLDivElement;
const saveAsPerson = document.getElementById('saveAsPerson') as HTMLDivElement;
const savePersonBtn = document.getElementById('savePersonBtn') as HTMLButtonElement;
const newPersonNameInput = document.getElementById('newPersonName') as HTMLInputElement;

interface PersonReference {
    id: string;
    thumb: string;
    detScore: number;
    addedAt: string;
}

interface PersonSummary {
    id: string;
    name: string;
    createdAt: string;
    references: PersonReference[];
}

let selectedPersonId: string | null = null;

async function loadPeople(): Promise<void> {
    let people: PersonSummary[];
    try {
        const res = await fetch('/people');
        if (!res.ok) throw new Error('bad status ' + res.status);
        people = await res.json();
    } catch (e) {
        console.warn('Could not load saved people', e);
        peoplePicker.innerHTML = '<em class="text-danger">Could not load saved people.</em>';
        // A failed reload must not leave a selection alive behind an error
        // message: the no-selection guard on submit would pass, /search
        // would receive a personId the server can't resolve, and a whole
        // directory scan would silently complete as "Matched 0" instead of
        // visibly failing.
        selectedPersonId = null;
        return;
    }

    // Same reasoning on the success path: a person selected in an earlier
    // load may have been deleted since (in another tab, or `data/` wiped).
    // If the id we're holding isn't among what the server just returned,
    // drop it -- the no-selection guard on submit is the only thing that
    // stands between a stale id and a silent zero-match scan.
    if (selectedPersonId && !people.some(p => p.id === selectedPersonId)) {
        selectedPersonId = null;
    }

    peoplePicker.innerHTML = people.length === 0
        ? '<em>No saved people yet. Upload a photo and save it.</em>'
        : '';
    for (const person of people) {
        const el = document.createElement('div');
        el.className = 'd-inline-block text-center align-top me-3 mb-2';
        el.style.width = '84px';
        el.innerHTML =
            `<img src="/people-files/${person.references[0].thumb}" width="64" height="64" ` +
            `class="person-avatar" style="cursor:pointer;border-radius:6px">` +
            `<div style="font-size:12px">${escapeHtml(person.name)}</div>` +
            `<div>` +
            `<button type="button" class="btn btn-sm btn-link p-0 add-ref-btn">+ photo</button> ` +
            `<button type="button" class="btn btn-sm btn-link p-0 manage-refs-btn">refs (${person.references.length})</button>` +
            `</div>` +
            `<button type="button" class="btn btn-sm btn-link text-danger p-0 delete-person-btn">delete</button>` +
            `<input type="file" accept=".jpg,.jpeg,.png,image/*" class="add-ref-input" style="display:none">` +
            `<div class="manage-refs-panel border rounded p-1 mt-1 text-start" style="display:none;font-size:11px"></div>`;

        const img = el.querySelector('img.person-avatar') as HTMLImageElement;
        if (person.id === selectedPersonId) {
            img.style.outline = '3px solid #0d6efd';
        }
        img.addEventListener('click', () => {
            selectedPersonId = person.id;
            document.querySelectorAll('#peoplePicker img.person-avatar')
                .forEach(i => ((i as HTMLElement).style.outline = ''));
            img.style.outline = '3px solid #0d6efd';
        });

        el.querySelector('.delete-person-btn')!.addEventListener('click', async () => {
            if (!confirm(`Delete ${person.name}?`)) return;
            try {
                const res = await fetch(`/people/${person.id}`, { method: 'DELETE' });
                if (!res.ok) { alert('Could not delete that person.'); return; }
            } catch (e) {
                alert('Could not reach the server.'); return;
            }
            if (selectedPersonId === person.id) selectedPersonId = null;
            await loadPeople();
        });

        // Adding another reference photo: reachable from the picker, per
        // the design spec (Part B) -- max-over-references is the whole
        // point of a saved person having more than one photo, and this is
        // the only way a user can ever add a second one.
        const addRefInput = el.querySelector('.add-ref-input') as HTMLInputElement;
        const addRefBtn = el.querySelector('.add-ref-btn') as HTMLButtonElement;
        addRefBtn.addEventListener('click', () => addRefInput.click());
        addRefInput.addEventListener('change', async () => {
            const file = addRefInput.files?.[0];
            if (!file) return;
            addRefBtn.disabled = true;
            try {
                const fd = new FormData();
                fd.append('photo', file);
                const res = await fetch(`/people/${person.id}/references`, { method: 'POST', body: fd });
                if (res.status === 422) { alert('No face found in that photo.'); return; }
                if (res.status === 404) {
                    alert('That person no longer exists. Refreshing the list.');
                    await loadPeople();
                    return;
                }
                if (!res.ok) { alert('Could not add that reference photo.'); return; }
                await loadPeople();
            } catch (e) {
                alert('Could not reach the server.');
            } finally {
                addRefBtn.disabled = false;
                addRefInput.value = '';
            }
        });

        // Managing references: a small inline panel per person, expanded on
        // demand, listing each reference photo with its own delete control.
        const manageBtn = el.querySelector('.manage-refs-btn') as HTMLButtonElement;
        const managePanel = el.querySelector('.manage-refs-panel') as HTMLDivElement;
        manageBtn.addEventListener('click', () => {
            const isOpen = managePanel.style.display !== 'none';
            if (isOpen) {
                managePanel.style.display = 'none';
                return;
            }
            managePanel.innerHTML = '';
            for (const ref of person.references) {
                const row = document.createElement('div');
                row.className = 'd-flex align-items-center justify-content-between mb-1';
                row.innerHTML =
                    `<img src="/people-files/${ref.thumb}" width="28" height="28" style="border-radius:4px">` +
                    `<button type="button" class="btn btn-sm btn-link text-danger p-0 ms-1">remove</button>`;
                row.querySelector('button')!.addEventListener('click', async () => {
                    if (!confirm('Remove this reference photo?')) return;
                    try {
                        const delRes = await fetch(
                            `/people/${person.id}/references/${ref.id}`, { method: 'DELETE' });
                        if (delRes.status === 409) {
                            alert('This is the only reference photo left. ' +
                                'Delete the person instead of removing it.');
                            return;
                        }
                        if (delRes.status === 404) {
                            alert('That reference no longer exists. Refreshing the list.');
                            await loadPeople();
                            return;
                        }
                        if (!delRes.ok) { alert('Could not remove that reference photo.'); return; }
                        await loadPeople();
                    } catch (e) {
                        alert('Could not reach the server.');
                    }
                });
                managePanel.appendChild(row);
            }
            managePanel.style.display = 'block';
        });

        peoplePicker.appendChild(el);
    }
}

function escapeHtml(s: string): string {
    const div = document.createElement('div');
    div.innerText = s;
    return div.innerHTML;
}

function sourceChanged() {
    const usingPerson = radioSourcePerson.checked;
    peoplePicker.style.display = usingPerson ? 'block' : 'none';
    sourceInput.style.display = usingPerson ? 'none' : 'block';
    saveAsPerson.style.display = 'none';
    sourceImg.src = '';
    cleanCanvases();
    if (usingPerson) {
        loadPeople();
    }
}
radioSourceUpload.addEventListener('change', sourceChanged);
radioSourcePerson.addEventListener('change', sourceChanged);

savePersonBtn.addEventListener('click', async () => {
    const name = newPersonNameInput.value.trim();
    const file = sourceInput.files?.[0];
    if (!name || !file) { alert('Pick a photo and enter a name.'); return; }
    savePersonBtn.disabled = true;
    try {
        const fd = new FormData();
        fd.append('photo', file);
        fd.append('name', name);
        const res = await fetch('/people', { method: 'POST', body: fd });
        if (res.status === 422) { alert('No face found in that photo.'); return; }
        if (!res.ok) { alert('Could not save that person.'); return; }
        alert(`Saved ${name}.`);
        newPersonNameInput.value = '';
    } catch (e) {
        alert('Could not reach the server.');
    } finally {
        savePersonBtn.disabled = false;
    }
});

// target selection
const radioTargetSingle = document.getElementById('targetTypeSingle') as HTMLInputElement;
const radioTargetDirectory = document.getElementById('targetTypeDirectory') as HTMLInputElement;
const targetSingleInput = document.getElementById('targetSingle') as HTMLInputElement;
const targetDirectoryInput = document.getElementById('targetDirectory') as HTMLInputElement;

// target preview
const progressContainer = document.getElementById('progress-container') as HTMLDivElement;
const targetImg = document.getElementById('targetImg') as HTMLImageElement;

radioTargetSingle.addEventListener('change', targetChanged);
radioTargetDirectory.addEventListener('change', targetChanged);

function targetChanged(event?: Event) {
    targetImg.src = '';
    targetSingleInput.style.display = 'none';
    targetSingleInput.value = '';
    targetDirectoryInput.style.display = 'none';
    targetDirectoryInput.value = '';
    progressContainer.style.display = 'none';
    if (radioTargetSingle.checked) {
        targetSingleInput.style.display = 'block';
    } else if (radioTargetDirectory.checked) {
        targetDirectoryInput.style.display = 'block';
        progressContainer.style.display = 'block';
        resetCounters(0);
    }
}

sourceInput.addEventListener('change', (event) => {
    const sourceFile = sourceInput?.files?.[0];
    if (sourceFile) {
        sourceImg.src = URL.createObjectURL(sourceFile);
        cleanCanvases();
        saveAsPerson.style.display = 'none';
    }
});

targetSingleInput.addEventListener('change', (event) => {
    const targetFile = targetSingleInput.files?.[0];
    if (targetFile) {
        targetImg.src = URL.createObjectURL(targetFile);
        cleanCanvases();
    }
});

targetDirectoryInput.addEventListener('change', (event) => {
    const targetFiles = targetDirectoryInput.files;
    if (targetFiles) {
        cleanCanvases();
    }
});

let sendingRequest = false;
let filesToBeZipped: File[] = [];
const counters = {
    total: 0,
    requestsSent: 0,
    responsesReceived: 0,
    filesMatched: 0,
    failed: 0,
}

const form = document.querySelector('form') as HTMLFormElement;
form.addEventListener('submit', async (event) => {
    event.preventDefault();

    if (sendingRequest) {
        alert('Previous request is still in progress, please wait.'); return;
    }

    const form = event.currentTarget as HTMLFormElement;
    const usingPerson = radioSourcePerson.checked;

    if (usingPerson && !selectedPersonId) {
        alert('Please choose a saved person.'); return;
    }
    const sourceFile = usingPerson ? undefined : sourceInput.files?.[0];
    if (!usingPerson && !sourceFile) {
        alert('Please select source image'); return;
    }

    // The directory branch below resets this via resetCounters(), but the
    // single-image branch never calls resetCounters -- without this, a
    // failure count from an earlier run would carry over silently.
    counters.failed = 0;

    const files = (isDirectory()?targetDirectoryInput:targetSingleInput).files!;
    const targetFiles = Array.from(files).filter(file => file.name.match(/.*\.(jpe?g|png)$/i));
    if (targetFiles.length == 0) {
        alert('Please select target image(s)'); return;
    }

    let source: SourceRef;
    if (usingPerson) {
        // Zero /probe calls in this mode - the saved person's embedding
        // already exists server-side, which is the entire point of saving
        // one in the first place.
        source = { personId: selectedPersonId! };
    } else {
        // One probe for the whole run - this is the entire point of the split.
        let probe: number[];
        try {
            const probeForm = new FormData();
            probeForm.append('source', sourceFile!);
            const probeRes = await fetch('/probe', { method: 'POST', body: probeForm });
            if (probeRes.status === 422) {
                alert('No face found in the source image. Try a clearer photo.');
                return;
            }
            if (!probeRes.ok) { alert('Could not read the source image.'); return; }
            const probeJson = await probeRes.json();
            probe = probeJson.embedding;
            drawSourceBox(probeJson.face.box);
            saveAsPerson.style.display = 'block';
        } catch (e) {
            alert('Could not reach the server.'); return;
        }
        source = { probe };
    }

    if (!isDirectory()) { // single file
        try {
            await sendRequest(source, targetFiles[0]);
            // Single-image mode has no progress panel to show a failure
            // count in -- an explicit alert is the only way this failure
            // wouldn't be entirely silent (no boxes drawn, nothing else
            // visible).
            if (counters.failed > 0) {
                alert('Could not read that image. Try again or pick a different photo.');
            }
        } finally {
            sendingRequest = false;
        }
    }
    else { // multiple files
        resetCounters(targetFiles.length);
        filesToBeZipped = [];
        async.eachLimit(targetFiles, 2, (targetFile: File, callback) => {
            // iterator couldn't be an async function
            sendRequest(source, targetFile)
                .then(() => callback())
                .catch((error) => callback(error));
        }, async (error) => {
            sendingRequest = false;
            if (error) {
                console.error(error);
            }
            else {
                console.log('Completed');
                if (filesToBeZipped.length > 0) {
                    // zip and download
                    const zip = new JSZip();
                    for (const file of filesToBeZipped) {
                        zip.file(file.name, file);
                    }
                    filesToBeZipped = []; // free up memory
                    const blob = await zip.generateAsync({type:'blob'});
                    const ts = new Date().toISOString()
                    saveAs(blob, 'download-'+ts+'.zip');
                }
            }
        });
    }
});

function isDirectory() {
    return (form.elements.namedItem('targetType') as RadioNodeList).value == 'directory';
}
function isSingle() {
    return !isDirectory();
}

type SourceRef = { probe: number[] } | { personId: string };

function sendRequest(source: SourceRef, targetFile: File): Promise<void> {
    onRequestSent();
    sendingRequest = true;
    return (async () => {
        try {
            const body = new FormData();
            body.append('target', targetFile);
            if ('personId' in source) {
                body.append('personId', source.personId);
            } else {
                body.append('probe', JSON.stringify([source.probe]));
            }
            const res = await fetch('/search', { method: 'POST', body });
            await handleResponse(res, targetFile);
        } catch (error) {
            // A network failure (or an unparseable response) for one photo
            // must not abort a directory scan or leave sendingRequest stuck
            // for a single-image run -- it's counted and skipped, same as
            // a non-2xx response from handleResponse. Counted here too, not
            // just logged: a run where every request fails at the
            // network/parse level (never reaching handleResponse's non-2xx
            // branch at all) must not still report a clean "found nobody".
            counters.failed += 1;
            console.warn(targetFile.name, 'search request failed', error);
        } finally {
            onResponseReceived();
        }
    })();
}

async function handleResponse(res: Response, targetFile: File): Promise<void> {
    const body = await res.json();
    if (!res.ok) {
        // Previously swallowed into console.warn alone -- a whole directory
        // scan could complete reporting "Matched 0 out of N" with every
        // single request having actually failed (e.g. a stale personId the
        // server rejects with 400 bad_probe), indistinguishable from
        // "nobody matched". Counted so displayProgress can surface it.
        counters.failed += 1;
        console.warn(targetFile.name, body.error);
        return;
    }

    if (isSingle()) {
        const targetCanvas = document.createElement('canvas');
        targetCanvas.id = 'targetCanvas';
        locateElementOnTopOf(targetImg, targetCanvas);
        for (const face of body.faces) {
            canvasRect(targetCanvas, face.box, face.matched ? '#FF0000' : '#888888');
            if (face.matched) {
                canvasRectLabel(targetCanvas, face.cosine.toFixed(3) + ' cosine', face.box);
            }
        }
        return;
    }

    if (body.matched) {
        onFaceMatched();
        filesToBeZipped.push(targetFile);
    }
}

function drawSourceBox(box: RelativeBox) {
    cleanCanvases();
    const sourceCanvas = document.createElement('canvas');
    sourceCanvas.id = 'sourceCanvas';
    locateElementOnTopOf(sourceImg, sourceCanvas);
    canvasRect(sourceCanvas, box, '#FF0000');
}

function cleanCanvases() {
    document.getElementById('sourceCanvas')?.remove();
    document.getElementById('targetCanvas')?.remove();
}

function locateElementOnTopOf(existingElement: HTMLImageElement, newElement: HTMLCanvasElement) {
    newElement.width = existingElement.width;
    newElement.height = existingElement.height;
    newElement.style.position = 'absolute';
    newElement.style.top = existingElement.offsetTop+'px';
    newElement.style.left = existingElement.offsetLeft+'px';
    existingElement.after(newElement);
}

type RelativeBox = {
    top: number,
    left: number,
    width: number,
    height: number
}

function canvasRect(canvas: HTMLCanvasElement, boundingBox: RelativeBox, strokeStyle?: string) {
    const context = canvas.getContext('2d');
    if (context) {
        context.strokeStyle = strokeStyle || '#FF0000';
        context.lineWidth = 2;
        context.strokeRect(
            boundingBox.left * canvas.width,
            boundingBox.top * canvas.height,
            boundingBox.width * canvas.width,
            boundingBox.height * canvas.height,
        );
    }
}

function canvasRectLabel(canvas: HTMLCanvasElement, text: string, rectBoundingBox: RelativeBox) {
    const context = canvas.getContext('2d');
    if (context) {
        context.fillStyle = '#FF0000';
        context.font = 'bold 19px Arial';
        context.shadowColor="black";
        context.shadowBlur=7;
        context.lineWidth = 2;
        context.fillText(
            text,
            rectBoundingBox.left * canvas.width,
            (rectBoundingBox.top * canvas.height) - 15,
        );
    }
}

function resetCounters(total: number) {
    counters.total = total;
    counters.requestsSent = 0;
    counters.responsesReceived = 0;
    counters.filesMatched = 0;
    counters.failed = 0;
    displayProgress();
}

function onRequestSent() {
    counters.requestsSent += 1;
    displayProgress();
}

function onResponseReceived() {
    counters.responsesReceived += 1;
    displayProgress();
}

function onFaceMatched() {
    counters.filesMatched += 1;
    displayProgress();
}

function displayProgress() {
    console.log(counters);
    const progressPercent = 
        counters.total > 0 ?
            Math.ceil((counters.responsesReceived/counters.total)*100) :
            0;
    const matchedPercent =
        counters.total > 0 ?
            Math.ceil((counters.filesMatched/counters.total)*100) :
            0;
    (document.getElementById('progress-percent') as HTMLDivElement).style.width = progressPercent+'%';
    (document.getElementById('progress-number') as HTMLDivElement).innerHTML =
        'Scanned '+counters.responsesReceived+' out of '+counters.total+' ('+progressPercent+'%)';
    (document.getElementById('progress-matched') as HTMLDivElement).innerHTML =
        'Matched '+counters.filesMatched+' out of '+counters.total+' ('+matchedPercent+'%)';

    // A scan that failed must not look like a scan that found nobody: shown
    // alongside (not instead of) the counts above, so "Matched 0" is never
    // the only thing visible when every request actually errored out.
    const errorsEl = document.getElementById('progress-errors') as HTMLDivElement;
    if (counters.failed > 0) {
        errorsEl.style.display = 'block';
        errorsEl.innerHTML = counters.failed+' request(s) failed -- results may be incomplete.';
    } else {
        errorsEl.style.display = 'none';
        errorsEl.innerHTML = '';
    }
}
