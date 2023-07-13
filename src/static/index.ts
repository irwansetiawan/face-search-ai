
const sourceInput = document.getElementById('source') as HTMLInputElement;
sourceInput?.addEventListener('change', (event) => {
    const sourceFile = sourceInput?.files?.[0];
    if (sourceFile) {
        console.log(sourceFile);
    }
});

const targetInput = document.getElementById('target') as HTMLInputElement;
targetInput?.addEventListener('change', (event) => {
    const targetFile = targetInput?.files?.[0];
    if (targetFile) {
        console.log(targetFile);
    }
});
