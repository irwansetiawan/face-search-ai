import * as async from 'async';

const radioTargetSingle = document.getElementById('targetTypeSingle') as HTMLInputElement;
const radioTargetDirectory = document.getElementById('targetTypeDirectory') as HTMLInputElement;
const targetSingle = document.getElementById('targetSingle') as HTMLInputElement;
const targetDirectory = document.getElementById('targetDirectory') as HTMLInputElement;

radioTargetSingle.addEventListener('change', targetChanged);
radioTargetDirectory.addEventListener('change', targetChanged);

function targetChanged(event?: Event) {
    targetSingle.style.display = 'none';
    targetDirectory.style.display = 'none';
    if (radioTargetSingle.checked) {
        targetSingle.style.display = 'block';
    } else if (radioTargetDirectory.checked) {
        targetDirectory.style.display = 'block';
    }
}

const sourceImg = document.getElementById('sourceImg') as HTMLImageElement;
const sourceInput = document.getElementById('source') as HTMLInputElement;
sourceInput.addEventListener('change', (event) => {
    const sourceFile = sourceInput?.files?.[0];
    if (sourceFile) {
        sourceImg.src = URL.createObjectURL(sourceFile);
        cleanCanvases();
    }
});

const targetImg = document.getElementById('targetImg') as HTMLImageElement;
const targetSingleInput = document.getElementById('targetSingle') as HTMLInputElement;
targetSingleInput.addEventListener('change', (event) => {
    const targetFile = targetSingleInput.files?.[0];
    if (targetFile) {
        targetImg.src = URL.createObjectURL(targetFile);
        cleanCanvases();
    }
});

const targetDirectoryInput = document.getElementById('targetDirectory') as HTMLInputElement;
targetDirectoryInput.addEventListener('change', (event) => {
    const targetFiles = targetDirectoryInput.files;
    if (targetFiles) {
        targetImg.src = '';
        cleanCanvases();
    }
});

const form = document.querySelector('form') as HTMLFormElement;
form.addEventListener('submit', (event) => {
    event.preventDefault();
    const form = event.currentTarget as HTMLFormElement;
    const sourceFile = (form.elements.namedItem('source') as HTMLInputElement).files?.[0];
    if (!sourceFile) {
        alert('Please select source image'); return;
    }

    const files = (form.elements.namedItem(isDirectory()?'targetDirectory':'targetSingle') as HTMLInputElement).files!;
    const targetFiles = Array.from(files).filter(file => file.name.match(/.*(.[jpe?g|png]$)/gi));
    if (targetFiles.length == 0) {
        alert('Please select target image'); return;
    }

    if (!isDirectory) { // single file
        sendRequest(form, sourceFile, targetFiles[0]);
    } else { // multiple files
        async.eachLimit(targetFiles, 2, (targetFile: File, callback) => {
            sendRequest(form, sourceFile, targetFile)
                .then((res) => callback())
                .catch((err) => callback(err));
        }, (error) => {
            if (error) console.error(error);
            else console.log('Completed');
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
    return new Promise((resolve, reject) => {
        console.log('Sending request for source file '+sourceFile.name+', and target file '+targetFile.name);
        const url = new URL(form.action);
        const fetchOptions: RequestInit = {
            method: form.method,
            body: getFormData(sourceFile, targetFile),
        };
        fetch(url, fetchOptions)
            .then((res) => {
                handleResponse(res);
                resolve(res);
            })
            .catch((error) => {
                const responseDiv = document.getElementById('response') as HTMLDivElement;
                responseDiv.innerHTML = error;
                reject(error);
            });
    })
}

async function handleResponse(res: Response) {
    const responseDiv = document.getElementById('response') as HTMLDivElement;
    const responseJson = JSON.parse(await res.text());
    responseDiv.innerHTML = JSON.stringify(responseJson, null, 4);

    const sourceImageFace = responseJson.SourceImageFace;
    const faceMatches = responseJson.FaceMatches;
    const unmatchedFaces = responseJson.UnmatchedFaces;

    cleanCanvases();
    // create a canvas on top of the source image
    const sourceCanvas = document.createElement('canvas') as HTMLCanvasElement;
    sourceCanvas.id = 'sourceCanvas';
    locateElementOnTopOf(sourceImg, sourceCanvas);
    canvasRect(sourceCanvas, sourceImageFace.BoundingBox, '#FF0000');

    if (isSingle()) {
        // create a canvas on top of the target image
        const targetCanvas = document.createElement('canvas') as HTMLCanvasElement;
        targetCanvas.id = 'targetCanvas';
        locateElementOnTopOf(targetImg, targetCanvas);
        for (const faceMatch of faceMatches) {
            canvasRect(targetCanvas, faceMatch.Face.BoundingBox, '#FF0000');
            canvasRectLabel(targetCanvas, faceMatch.Similarity.toFixed(1)+'% similarity', faceMatch.Face.BoundingBox)
        }
    }
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