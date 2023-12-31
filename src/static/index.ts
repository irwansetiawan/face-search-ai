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

    if (!isDirectory) { // single file
        await sendRequest(form, sourceFile, targetFiles[0]);
        sendingRequest = false;
    }
    else { // multiple files
        resetCounters(targetFiles.length);
        filesToBeZipped = [];
        async.eachLimit(targetFiles, 2, (targetFile: File, callback) => {
            // iterator couldn't be an async function
            sendRequest(form, sourceFile, targetFile)
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

function getFormData(sourceFile: File, targetFile: File): FormData {
    const formData = new FormData();
    formData.append('source', sourceFile);
    formData.append('target', targetFile);
    return formData;
}

function sendRequest(form: HTMLFormElement, sourceFile: File, targetFile: File): Promise<Response> {
    onRequestSent();
    sendingRequest = true;
    return new Promise(async (resolve, reject) => {
        console.log('Sending request for source file '+sourceFile.name+', and target file '+targetFile.name);
        const url = new URL(form.action);
        const fetchOptions: RequestInit = {
            method: form.method,
            body: getFormData(sourceFile, targetFile),
        };
        try {
            const res = await fetch(url, fetchOptions)
            onResponseReceived();
            await handleResponse(res, targetFile);
            resolve(res);
        } catch(error: any) {
            const responseDiv = document.getElementById('response') as HTMLDivElement;
            responseDiv.innerHTML = error;
            reject(error);
        }
    })
}

function handleResponse(res: Response, targetFile: File): Promise<void> {
    return new Promise(async (resolve, reject) => {
        const responseDiv = document.getElementById('response') as HTMLDivElement;
        const responseJson = JSON.parse(await res.text());
        responseDiv.innerHTML = JSON.stringify(responseJson, null, 4);
    
        const sourceImageFace = responseJson.SourceImageFace;
        const faceMatches = responseJson.FaceMatches;
        const unmatchedFaces = responseJson.UnmatchedFaces;
    
        cleanCanvases();
        if (sourceImageFace && sourceImageFace.BoundingBox) {
            // create a canvas on top of the source image
            const sourceCanvas = document.createElement('canvas') as HTMLCanvasElement;
            sourceCanvas.id = 'sourceCanvas';
            locateElementOnTopOf(sourceImg, sourceCanvas);
            canvasRect(sourceCanvas, sourceImageFace.BoundingBox, '#FF0000');
        }
    
        if (isSingle()) {
            if (faceMatches && faceMatches.length > 0) {
                // create a canvas on top of the target image
                const targetCanvas = document.createElement('canvas') as HTMLCanvasElement;
                targetCanvas.id = 'targetCanvas';
                locateElementOnTopOf(targetImg, targetCanvas);
                for (const faceMatch of faceMatches) {
                    canvasRect(targetCanvas, faceMatch.Face.BoundingBox, '#FF0000');
                    canvasRectLabel(targetCanvas, faceMatch.Similarity.toFixed(1)+'% similarity', faceMatch.Face.BoundingBox)
                }
            }
            resolve();
        }
        else {
            if (faceMatches && faceMatches.length > 0) {
                console.log(targetFile.name+' matches');
                onFaceMatched();
                filesToBeZipped.push(targetFile);
            } else {
                console.log(targetFile.name+' no match')
            }
            resolve();
        }
    })
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
    Top: number,
    Left: number,
    Width: number,
    Height: number
}

function canvasRect(canvas: HTMLCanvasElement, boundingBox: RelativeBox, strokeStyle?: string) {
    const context = canvas.getContext('2d');
    if (context) {
        context.strokeStyle = strokeStyle || '#FF0000';
        context.lineWidth = 2;
        context.strokeRect(
            boundingBox.Left * canvas.width,
            boundingBox.Top * canvas.height,
            boundingBox.Width * canvas.width,
            boundingBox.Height * canvas.height,
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
            rectBoundingBox.Left * canvas.width,
            (rectBoundingBox.Top * canvas.height) - 15,
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
    (document.getElementById('progress-percent') as HTMLDivElement).style.width = progressPercent+'%';
    (document.getElementById('progress-number') as HTMLDivElement).innerHTML = progressPercent+'%';
}
