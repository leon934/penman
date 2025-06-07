import { StateNode } from "tldraw";

export class ScreenshotPointing extends StateNode {
    static override id = "pointing";

    override onPointerMove() {
        if(this.editor.inputs.isDragging) {
            this.parent.transition("dragging");
        }
    }

    override onPointerUp() {
        this.complete();
    }

    override onCancel() {
        this.complete();
    }

    private complete() {
        this.parent.transition("idle");
    }
}