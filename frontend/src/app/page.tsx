"use client";

import { ScreenshotDragging } from '~/components/screenshotTool/Dragging';
import { ScreenshotTool } from '~/components/screenshotTool/ScreenshotTool';
import { 
    Tldraw,
    Box,
    DefaultToolbar,
    DefaultToolbarContent,
    type TLComponents,
    type TLUiAssetUrlOverrides,
    type TLUiOverrides,
    TldrawUiMenuItem,
    useEditor,
    useIsToolSelected,
    useTools, 
    useValue
} from 'tldraw'
import 'tldraw/tldraw.css'
import { Copy } from '~/components/Copy';

const customTools = [ScreenshotTool]

const customUiOverrides: TLUiOverrides = {
    tools: (editor, tools) => {
        return {
            ...tools,
            screenshot: {
                id: "screenshot",
                label: "Screenshot",
                icon: "tool-screenshot",
                kbd: "j",
                onSelect() {
                    editor.setCurrentTool("screenshot");
                }
            }
        }
    }
}

function CustomToolbar() {
    const tools = useTools()
    const isScreenshotSelected = useIsToolSelected(tools["screenshot"])
    const screenshotTool = tools['screenshot']

    return (
        <DefaultToolbar>
            {screenshotTool?.id && (
                <TldrawUiMenuItem
                    {...screenshotTool}
                    // We spread the rest of the tool props only AFTER confirming it's valid
                    id={screenshotTool.id} // Explicitly pass the now-guaranteed 'id'
                    isSelected={isScreenshotSelected}
                />
            )}
            <DefaultToolbarContent />
        </DefaultToolbar>
    )
}

const customAssetUrls: TLUiAssetUrlOverrides = {
    icons: {
        'tool-screenshot': '/penman_logo.svg'
    }
}

// Shows the screenshot box whenever the tool is being used and dragged.
function ScreenshotBox() {
    const editor = useEditor()

    const screenshotBrush = useValue(
        'screenshot brush',
        () => {
            if(editor.getPath() !== 'screenshot.dragging') return null

            const draggingState = editor.getStateDescendant<ScreenshotDragging>("screenshot.dragging")!

            const box = draggingState.screenshotBox.get()

            const zoomLevel = editor.getZoomLevel()
            const { x, y } = editor.pageToViewport({ x: box.x, y: box.y })

            return new Box(x, y, box.w * zoomLevel, box.h * zoomLevel)
        },
        [editor]
    )

    if(!screenshotBrush) return null

    return (
        <div 
            className="absolute top-0 left-0 border-2 border-dashed z-0"
            style={{
				transform: `translate(${screenshotBrush.x}px, ${screenshotBrush.y}px)`,
				width: screenshotBrush.w,
				height: screenshotBrush.h,
            }}
        />
    )
}

const customComponents: TLComponents = {
    InFrontOfTheCanvas: ScreenshotBox,
    Toolbar: CustomToolbar
}

export default function HomePage() {
    return (
        <div style={{ position: 'fixed', inset: 0}}>
            {/* TODO: Potentially replace persistenceKey argument with custom defined project name? */}
            <Tldraw
                persistenceKey='placeholder_user'
                tools={customTools}
                overrides={customUiOverrides}
                assetUrls={customAssetUrls}
                components={customComponents}
            >
                <Copy />
            </Tldraw>
        </div>
    );
}