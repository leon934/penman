import { StateNode, TLCancelEvent, TLInterruptEvent } from 'tldraw'
import { ScreenshotDragging } from './Dragging'
import { ScreenshotIdle } from './Idle'
import { ScreenshotPointing } from './Pointing'

export class ScreenshotTool extends StateNode {
	static override id = 'screenshot'
	static override initial = 'idle'
	static override children = () => [ScreenshotIdle, ScreenshotPointing, ScreenshotDragging]

	override onEnter = () => {
		this.editor.setCursor({ type: 'cross', rotation: 0 })
	}

	override onExit = () => {
		this.editor.setCursor({ type: 'default', rotation: 0 })
	}

	override onInterrupt: TLInterruptEvent = () => {
		this.complete()
	}

	override onCancel: TLCancelEvent = () => {
		this.complete()
	}

	private complete() {
		this.parent.transition('select', {})
	}
}