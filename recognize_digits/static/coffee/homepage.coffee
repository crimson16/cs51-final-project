# Functions for the Canvas


DrawCanvas = (canvasWidth, canvasHeight, canvasID) ->
    # Making the elements
    canvasDiv = document.getElementById("canvasDiv")
    canvas = document.createElement('canvas')
    canvas.setAttribute 'width', "500px" #canvasWidth
    canvas.setAttribute 'height', "500px" #canvasHeight
    canvas.setAttribute 'id', 'canvas'
    canvasDiv.appendChild canvas
    if typeof G_vmlCanvasManager != 'undefined'
        canvas = G_vmlCanvasManager.initElement(canvas)
    context = canvas.getContext('2d')

    # Mouse Down Event
    $('#canvas').mousedown (e) ->
        mouseX = e.pageX - @offsetLeft
        mouseY = e.pageY - @offsetTop
        paint = true
        addClick e.pageX - @offsetLeft, e.pageY - @offsetTop
        redraw()
        # return

    # Mouse Move Event
    $('#canvas').mousemove (e) ->
        if paint
            addClick e.pageX - @offsetLeft, e.pageY - @offsetTop, true
            redraw()
        # return

    # Mouse Up Event
    $('#canvas').mouseup (e) ->
        paint = false
        # return

    # Mouse Leave Event
    $('#canvas').mouseleave (e) ->
        paint = false
        # return

    # addClick function to save the click position:
    clickX = new Array
    clickY = new Array
    clickDrag = new Array
    paint = undefined

    addClick = (x, y, dragging) ->
        clickX.push x
        clickY.push y
        clickDrag.push dragging
        # return

    redraw = ->
        context.clearRect 0, 0, context.canvas.width, context.canvas.height
        # Clears the canvas
        context.strokeStyle = '#df4b26'
        context.lineJoin = 'round'
        context.lineWidth = 5
        i = 0
        while i < clickX.length
            context.beginPath()
            if clickDrag[i] and i
                context.moveTo clickX[i - 1], clickY[i - 1]
            else
                context.moveTo clickX[i] - 1, clickY[i]
            context.lineTo clickX[i], clickY[i]
            context.closePath()
            context.stroke()
            i++
        # return

$(document).ready(() -> 

    DrawCanvas(500, 500, "canvasDiv")

)
