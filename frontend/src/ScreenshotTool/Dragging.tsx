import { Box, StateNode, atom, exportToBlob } from 'tldraw'

export class ScreenshotDragging extends StateNode {
	static override id = 'dragging'

	screenshotBox = atom('screenshot brush', new Box())

	override onEnter = () => {
		this.update()
	}

	override onPointerMove = () => {
		this.update()
	}

	override onKeyDown = () => {
		this.update()
	}

	override onKeyUp = () => {
		this.update()
	}

	private update() {
		const {
			inputs: { shiftKey, altKey, originPagePoint, currentPagePoint },
		} = this.editor

		const box = Box.FromPoints([originPagePoint, currentPagePoint])

		if (shiftKey) {
			if (box.w > box.h * (16 / 9)) {
				box.h = box.w * (9 / 16)
			} else {
				box.w = box.h * (16 / 9)
			}

			if (currentPagePoint.x < originPagePoint.x) {
				box.x = originPagePoint.x - box.w
			}

			if (currentPagePoint.y < originPagePoint.y) {
				box.y = originPagePoint.y - box.h
			}
		}

		if (altKey) {
			box.w *= 2
			box.h *= 2
			box.x = originPagePoint.x - box.w / 2
			box.y = originPagePoint.y - box.h / 2
		}

		this.screenshotBox.set(box)
	}

	override onPointerUp = async () => {
		const { editor } = this
		const box = this.screenshotBox.get()

		const shapes = editor.getCurrentPageShapes().filter((s) => {
			const pageBounds = editor.getShapeMaskedPageBounds(s)
			if (!pageBounds) return false
			return box.includes(pageBounds)
		})

		if (shapes.length) {
			const blob = await exportToBlob({
				editor,
				ids: shapes.map((s) => s.id),
				format: 'png',
				opts: { background: true },
			});

			const formData = new FormData();
			formData.append('file', blob, 'Screenshot.png');

			const response = await fetch('http://localhost:5000/image', {
				method: 'POST',
				body: formData,
				headers: {
					'Accept': 'application/json',
				},
			});

			if (response.ok) {
				console.log("Successfully sent to backend.");
			} else {
				console.log("Failed to send to backend. ", response.statusText);
			}
		}

		this.editor.setCurrentTool('select')
	}

	override onCancel = () => {
		this.editor.setCurrentTool('select')
	}
}