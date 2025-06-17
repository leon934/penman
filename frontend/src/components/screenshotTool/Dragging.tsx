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

        console.log(shapes)

        if(shapes.length) {
            try {
                // Converts image to data url to be sent to the backend.
                const screenshot = await editor.toImage(shapes);
                const dataURL = await blobToDataURL(screenshot.blob);
                const payload = {
                    imageData: dataURL
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

                console.log(0)

                this.editor.setCurrentTool('select')

                const response = await fetch(apiURL, request);

                if(!response.ok) { throw new Error(`Server responded with status when attempting to : ${response.status}`) };

                // Handle response's json.
                response.json().then((data) => {
                    // console.log(data.predicted_value);
                    // console.log(data.confidence);

                    const tldraw_number = data.tldraw_number

                    let furthestX = 0
                    let furthestShape = undefined;

                    for(const shape of shapes) {
                        console.log(shape.x)

                        if(shape?.x === undefined) continue;
                        console.log(1)

                        furthestX = Math.max(furthestX, shape.x);
                        furthestShape = shape;
                    }

                    if(!furthestShape) {
                        console.log("Unable to find furthest shape. Returning.")
                        return
                    }

                    // const coordinates = tldraw_number.props.segments[0].points

                    // let maxX = 0
                    // for(const coordinate of coordinates) {
                    //     maxX = Math.max(maxX, coordinate.x)
                    // }

                    this.editor.createShape({
                        type: furthestShape.type,
                        x: furthestShape.x + 100,
                        y: furthestShape.y,
                        props: tldraw_number.props
                    })

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