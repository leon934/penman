import { StateNode, Box, atom } from "tldraw";

function blobToDataURL(blob: Blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        // Tells the program that reader.result is always a string.
        reader.onload = () => {
            if(typeof reader.result === "string") {
                resolve(reader.result)
            } else {
                reject(new Error("Expected a string."))
            }
        };
        reader.onerror = () => reject(reader.error);

        reader.readAsDataURL(blob);
    })
}

export class ScreenshotDragging extends StateNode {
    static override id = "dragging";

    screenshotBox = atom('screenshot brush', new Box())

    override onEnter() {
        this.update();
    }

    override onPointerMove() {
        this.update();
    }

    private update() {
        const {
            inputs: {originPagePoint, currentPagePoint}
        } = this.editor

        const box = Box.FromPoints([originPagePoint, currentPagePoint]);

        this.screenshotBox.set(box);
    }

    override onPointerUp = async () => {
        const { editor } = this;
        const box = this.screenshotBox.get()

        const shapes = editor.getCurrentPageShapes().filter((s) => {
            const pageBounds = editor.getShapeMaskedPageBounds(s);

            if(!pageBounds) return false;

            return box.includes(pageBounds);
        })

        if(shapes.length) {
            try {
                const screenshot = await editor.toImage(shapes);
                const dataURL = await blobToDataURL(screenshot.blob);
                const payload = {
                    imageData: dataURL,
                    // TODO: Replace this with whatever in the future as an identifier.
                    filename: "placeholder_screenshot"
                }

                const apiURL = 'http://127.0.0.1:5000/image'
                const request = {
                    method: "POST",
                    headers: {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                }

                this.editor.setCurrentTool('select')

                const response = await fetch(apiURL, request);

                if(!response.ok) { throw new Error(`Server responded with status: ${response.status}`) };

                console.log("Server responded successfully.");
            } catch(e) {
                console.log("Error occured", e)
                this.editor.setCurrentTool('select')
            }
            
        }

        this.editor.setCurrentTool('select')
    }

    override onCancel() {
        this.editor.setCurrentTool('select');
    }
}