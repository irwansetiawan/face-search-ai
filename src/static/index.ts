import * as async from 'async';
import { saveAs } from 'file-saver';
import JSZip from 'jszip';

// source image
const sourceInput = document.getElementById('source') as HTMLInputElement;

// source preview
const sourceImg = document.getElementById('sourceImg') as HTMLImageElement;

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
}

const form = document.querySelector('form') as HTMLFormElement;
form.addEventListener('submit', async (event) => {
    event.preventDefault();

    if (sendingRequest) {
        alert('Previous request is still in progress, please wait.'); return;
    }

    const form = event.currentTarget as HTMLFormElement;
    const sourceFile = sourceInput.files?.[0];
    if (!sourceFile) {
        alert('Please select source image'); return;
    }

    const files = (isDirectory()?targetDirectoryInput:targetSingleInput).files!;
    const targetFiles = Array.from(files).filter(file => file.name.match(/.*\.(jpe?g|png)$/i));
    if (targetFiles.length == 0) {
        alert('Please select target image(s)'); return;
    }

    // One probe for the whole run - this is the entire point of the split.
    let probe: number[];
    try {
        const probeForm = new FormData();
        probeForm.append('source', sourceFile);
        const probeRes = await fetch('/probe', { method: 'POST', body: probeForm });
        if (probeRes.status === 422) {
            alert('No face found in the source image. Try a clearer photo.');
            return;
        }
        if (!probeRes.ok) { alert('Could not read the source image.'); return; }
        const probeJson = await probeRes.json();
        probe = probeJson.embedding;
        drawSourceBox(probeJson.face.box);
    } catch (e) {
        alert('Could not reach the server.'); return;
    }

    if (!isDirectory()) { // single file
        try {
            await sendRequest(probe, targetFiles[0]);
        } finally {
            sendingRequest = false;
        }
    }
    else { // multiple files
        resetCounters(targetFiles.length);
        filesToBeZipped = [];
        async.eachLimit(targetFiles, 2, (targetFile: File, callback) => {
            // iterator couldn't be an async function
            sendRequest(probe, targetFile)
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

function sendRequest(probe: number[], targetFile: File): Promise<void> {
    onRequestSent();
    sendingRequest = true;
    return (async () => {
        try {
            const body = new FormData();
            body.append('target', targetFile);
            body.append('probe', JSON.stringify([probe]));
            const res = await fetch('/search', { method: 'POST', body });
            await handleResponse(res, targetFile);
        } catch (error) {
            // A network failure (or an unparseable response) for one photo
            // must not abort a directory scan or leave sendingRequest stuck
            // for a single-image run -- it's counted and skipped, same as
            // a non-2xx response from handleResponse.
            console.warn(targetFile.name, 'search request failed', error);
        } finally {
            onResponseReceived();
        }
    })();
}

async function handleResponse(res: Response, targetFile: File): Promise<void> {
    const body = await res.json();
    if (!res.ok) { console.warn(targetFile.name, body.error); return; }

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
}
