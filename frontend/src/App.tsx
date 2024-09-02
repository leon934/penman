import React from 'react';
import { ScreenshotTool } from './ScreenshotTool/ScreenshotTool';
import { ScreenshotDragging } from './ScreenshotTool/Dragging';
import {
	Box,
	DefaultToolbar,
	DefaultToolbarContent,
	TLComponents,
	TLUiAssetUrlOverrides,
	TLUiOverrides,
	Tldraw,
	TldrawUiMenuItem,
	useEditor,
	useIsToolSelected,
	useTools,
	useValue,
    TLUiComponents
} from 'tldraw'

const customTools = [ScreenshotTool]

const customUiOverrides: TLUiOverrides = {
	tools: (editor, tools) => {
		return {
			...tools,
			screenshot: {
				id: 'screenshot',
				label: 'Screenshot',
				icon: 'tool-screenshot',
				kbd: 'j',
				onSelect() {
					editor.setCurrentTool('screenshot')
				},
			},
		}
	},
}

function CustomToolbar() {
	const tools = useTools()
	const isScreenshotSelected = useIsToolSelected(tools['screenshot'])
	return (
		<DefaultToolbar>
			<TldrawUiMenuItem {...tools['screenshot']} isSelected={isScreenshotSelected} />
			<DefaultToolbarContent />
		</DefaultToolbar>
	)
}

const customAssetUrls: TLUiAssetUrlOverrides = {
	icons: {
		'tool-screenshot': '/penman_logo.svg',
	},
}

function ScreenshotBox() {
	const editor = useEditor()

	const screenshotBrush = useValue(
		'screenshot brush',
		() => {
			if (editor.getPath() !== 'screenshot.dragging') return null

			const draggingState = editor.getStateDescendant<ScreenshotDragging>('screenshot.dragging')!

			const box = draggingState.screenshotBox.get()

			const zoomLevel = editor.getZoomLevel()
			const { x, y } = editor.pageToViewport({ x: box.x, y: box.y })
			return new Box(x, y, box.w * zoomLevel, box.h * zoomLevel)
		},
		[editor]
	)

	if (!screenshotBrush) return null

	return (
		<div
			style={{
				position: 'absolute',
				top: 0,
				left: 0,
				transform: `translate(${screenshotBrush.x}px, ${screenshotBrush.y}px)`,
				width: screenshotBrush.w,
				height: screenshotBrush.h,
				border: '1px solid var(--color-text-0)',
				zIndex: 999,
			}}
		/>
	)
}

const customComponents: TLComponents = {
	InFrontOfTheCanvas: ScreenshotBox,
	Toolbar: CustomToolbar,
}

const components: Partial<TLUiComponents> = {
    ContextMenu: null,
	ActionsMenu: null,
	HelpMenu: null,
	ZoomMenu: null,
	MainMenu: null,
	Minimap: null,
	StylePanel: null, // Top right panel.
	PageMenu: null,
	// NavigationPanel: null,
	// Toolbar: null, // Bottom middle tool bar.
	// KeyboardShortcutsDialog: null,
	// QuickActions: null,
	// HelperButtons: null,
	DebugPanel: null,
	DebugMenu: null,
	SharePanel: null,
	// MenuPanel: null,
	// TopPanel: null,
	// CursorChatBubble: null,
}

const mergedComponents = {
    ...components,
    ...customComponents
};

const App: React.FC = () => {
    return (
        <div style={{ position: 'fixed', inset: 0 }}>
		    <Tldraw 
                components={mergedComponents} 
                tools={customTools}
                overrides={customUiOverrides}
                assetUrls={customAssetUrls}
            />
	    </div>
    );
}

export default App;
