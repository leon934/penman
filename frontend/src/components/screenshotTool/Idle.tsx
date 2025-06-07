import { StateNode } from "tldraw";

export class ScreenshotIdle extends StateNode {
    static override id = "idle";

    // Whenever we are in the idle state (after selecting the screenshot tool), we change to the "pointing" state.
    override onPointerDown() {
        this.parent.transition("pointing");
    }
}