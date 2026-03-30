const fs = require('fs');
const path = require('path');

const src = path.join(__dirname, '../../node_modules/onnxruntime-web/dist');
const dst = path.join(__dirname, '../../dist/onnx-dist');

if (!fs.existsSync(dst)) fs.mkdirSync(dst, { recursive: true });

let copied = 0;
for (const f of fs.readdirSync(src)) {
    if (f.endsWith('.wasm')) {
        fs.copyFileSync(path.join(src, f), path.join(dst, f));
        console.log(`Copied: ${f}`);
        copied++;
    }
}

console.log(`WASM files copied to dist/onnx-dist/ (${copied} files)`);