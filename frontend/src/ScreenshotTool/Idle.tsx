import { StateNode, TLEventHandlers } from 'tldraw'

export class ScreenshotIdle extends StateNode {
	static override id = 'idle'

	override onPointerDown: TLEventHandlers['onPointerUp'] = () => {
		this.parent.transition('pointing')
	}
}
