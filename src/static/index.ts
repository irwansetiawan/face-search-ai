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
targetChanged();

const sourceImg = document.getElementById('sourceImg') as HTMLImageElement;
const sourceInput = document.getElementById('source') as HTMLInputElement;
sourceImg.addEventListener('load', (event) => {
    const thisImg = event.currentTarget as HTMLImageElement;
    console.log('Image size: ', thisImg.clientWidth, thisImg.clientHeight)
});
sourceInput.addEventListener('change', (event) => {
    const sourceFile = sourceInput?.files?.[0];
    if (sourceFile) {
        sourceImg.src = URL.createObjectURL(sourceFile);

        sourceImg.nextElementSibling?.remove(); // remove all next siblings i.e. canvases
        targetImg.nextElementSibling?.remove(); // remove all next siblings i.e. canvases
    }
});

const targetImg = document.getElementById('targetImg') as HTMLImageElement;
const targetSingleInput = document.getElementById('targetSingle') as HTMLInputElement;
targetImg.addEventListener('load', (event) => {
    const thisImg = event.currentTarget as HTMLImageElement;
    console.log('Image size: ', thisImg.clientWidth, thisImg.clientHeight)
});
targetSingleInput.addEventListener('change', (event) => {
    const targetFile = targetSingleInput.files?.[0];
    if (targetFile) {
        targetImg.src = URL.createObjectURL(targetFile);

        sourceImg.nextElementSibling?.remove(); // remove all next siblings i.e. canvases
        targetImg.nextElementSibling?.remove(); // remove all next siblings i.e. canvases
    }
});

const targetDirectoryInput = document.getElementById('targetDirectory') as HTMLInputElement;
targetDirectoryInput.addEventListener('change', (event) => {
    const targetFiles = targetDirectoryInput.files;
    if (targetFiles) {
        console.log(targetFiles);

        sourceImg.nextElementSibling?.remove(); // remove all next siblings i.e. canvases
        targetImg.nextElementSibling?.remove(); // remove all next siblings i.e. canvases
    }
});

const form = document.querySelector('form') as HTMLFormElement;
form.addEventListener('submit', (event) => {
    const form = event.currentTarget as HTMLFormElement;
    const url = new URL(form.action);
    const formData = new FormData(form);
    const fetchOptions: RequestInit = {
        method: form.method,
        body: formData,
    };
    const responseDiv = document.getElementById('response') as HTMLDivElement;
    fetch(url, fetchOptions)
        .then(async (res) => {
            const responseJson = JSON.parse(await res.text());
            responseDiv.innerHTML = JSON.stringify(responseJson, null, 4);

            const sourceImageFace = responseJson.SourceImageFace;
            const faceMatches = responseJson.FaceMatches;
            const unmatchedFaces = responseJson.UnmatchedFaces;

            // create a canvas on top of the source image
            const sourceCanvas = document.createElement('canvas') as HTMLCanvasElement;
            locateElementOnTopOf(sourceImg, sourceCanvas);
            canvasRect(sourceCanvas, sourceImageFace.BoundingBox, '#FF0000');

            // create a canvas on top of the target image
            const targetCanvas = document.createElement('canvas') as HTMLCanvasElement;
            locateElementOnTopOf(targetImg, targetCanvas);
            for (const faceMatch of faceMatches) {
                canvasRect(targetCanvas, faceMatch.Face.BoundingBox, '#FF0000');
                canvasRectLabel(targetCanvas, faceMatch.Similarity.toFixed(1)+'% similarity', faceMatch.Face.BoundingBox)
            }
        })
        .catch((error) => {
            responseDiv.innerHTML = error;
        });
    event.preventDefault();
});

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