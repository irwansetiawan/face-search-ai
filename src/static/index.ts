const sourceImg = document.getElementById('sourceImg') as HTMLImageElement;
const sourceInput = document.getElementById('source') as HTMLInputElement;
sourceImg.addEventListener('load', (event) => {
    const thisImg = event.currentTarget as HTMLImageElement;
    console.log('Image size: ', thisImg.clientWidth, thisImg.clientHeight)
});
sourceInput?.addEventListener('change', (event) => {
    const sourceFile = sourceInput?.files?.[0];
    if (sourceFile) {
        sourceImg.src = URL.createObjectURL(sourceFile);
    }
});

const targetImg = document.getElementById('targetImg') as HTMLImageElement;
const targetInput = document.getElementById('target') as HTMLInputElement;
targetImg.addEventListener('load', (event) => {
    const thisImg = event.currentTarget as HTMLImageElement;
    console.log('Image size: ', thisImg.clientWidth, thisImg.clientHeight)
});
targetInput?.addEventListener('change', (event) => {
    const targetFile = targetInput?.files?.[0];
    if (targetFile) {
        targetImg.src = URL.createObjectURL(targetFile);
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
        })
        .catch((error) => {
            responseDiv.innerHTML = error;
        });
    event.preventDefault();
});
