import { StateNode, Box, atom, type TLDrawShape } from "tldraw";
import { mathTokens } from "../TLDrawMathTokens";

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
                // Converts image to data url to be sent to the backend.
                const screenshot = await editor.toImage(shapes);
                const dataURL = await blobToDataURL(screenshot.blob);
                const payload = {
                    imageData: dataURL
                }

                const apiURL = `${process.env.NEXT_PUBLIC_API_URL}/api/predict`
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

                if(!response.ok) { throw new Error(`Server responded with status when attempting to : ${response.status}`) };

                response.json().then((data) => {
                    let tldrawNumbers: TLDrawShape[] = []

                    console.log(data.determined_value, data.predicted_values)

                    for(const token of data.determined_value) {
                        const shape = mathTokens[token as any]
                        if(shape) tldrawNumbers.push(shape)
                    }
                    
                    let furthestX = 0
                    let furthestShape = undefined

                    for(const shape of shapes) {
                        furthestX = Math.max(furthestX, shape.x);
                        furthestShape = shape;
                    }

                    if(!furthestShape) {
                        console.log("Unable to find furthest shape. Returning.")
                        return
                    }
                    
                    // TODO: The offset is how much to shift the number by. Depending on the usecase, the offset would lead to overlapping or spread-apart math tokens.
                    const offset = 100

                    // Creates a new shape and offsets them.
                    const newShapes = Object.entries(tldrawNumbers).map(([i, val]) => {
                        const shape = Array.isArray(val) ? val[0] : val;

                        if(!val || typeof val !== 'object' || !shape.props || !shape.type) {
                            console.log(`Invalid shape at index ${i}`, val)
                            console.log(shape)
                        }
                        
                        return {
                            props: shape.props,
                            type: shape.type,
                            x: furthestShape.x + Number(i) * offset + offset,
                            y: furthestShape.y
                        };
                    })

                    this.editor.createShapes(newShapes)
                    console.log("Result drawn successfully.")
                })
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