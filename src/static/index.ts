const sourceImg = document.getElementById('sourceImg') as HTMLImageElement;
const sourceInput = document.getElementById('source') as HTMLInputElement;
sourceInput?.addEventListener('change', (event) => {
    const sourceFile = sourceInput?.files?.[0];
    if (sourceFile) {
        sourceImg.src = URL.createObjectURL(sourceFile);
    }
});

const targetImg = document.getElementById('targetImg') as HTMLImageElement;
const targetInput = document.getElementById('target') as HTMLInputElement;
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
    const fetchOptions = {
        method: form.method,
        body: formData,
    };
    fetch(url, fetchOptions);
    event.preventDefault();
});
