"use client"

import { useEffect } from 'react'
import { useEditor } from 'tldraw'

export const Copy = () => {
    const editor = useEditor()

    // FIXME: Remove this code after fully fleshing out creating output numbers.
    useEffect(() => {
        if(!editor) return

        const onCopy = () => {
            const selectedShape = editor.getSelectedShapes()[0]
            
            if(!selectedShape) return

            const newShape = {
                type: "draw",
                x: selectedShape.x + 100,
                y: selectedShape.y,
                props: selectedShape.props
            }

            editor.createShape(newShape);
            
            const json = JSON.stringify(newShape, null, 4);
            const blob = new Blob([json], { type: "application/json" })
            const url = URL.createObjectURL(blob)

            const a = document.createElement('a')
            a.href = url

            const number = 1

            a.download = `${number}.json`
            a.click()

            URL.revokeObjectURL(url)
        }

        window.addEventListener('copy', onCopy);

        return () => window.removeEventListener('copy', onCopy)
    }, [editor])

    return null
}