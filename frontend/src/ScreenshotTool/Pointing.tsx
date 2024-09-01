import { StateNode, TLEventHandlers } from 'tldraw'

export class ScreenshotPointing extends StateNode {
	static override id = 'pointing'

	override onPointerMove: TLEventHandlers['onPointerUp'] = () => {
		if (this.editor.inputs.isDragging) {
			this.parent.transition('dragging')
		}
	}

	override onPointerUp: TLEventHandlers['onPointerUp'] = () => {
		this.complete()
	}

	override onCancel: TLEventHandlers['onCancel'] = () => {
		this.complete()
	}

	private complete() {
		this.parent.transition('idle')
	}
}