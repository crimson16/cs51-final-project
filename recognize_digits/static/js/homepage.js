  
function makeCanvas (canvasID) {

  var canvasWidth = "280px"
  var canvasHeight = "280px"


  var canvasDiv = document.getElementById(canvasID);
  canvas = document.createElement('canvas');
  canvas.setAttribute('width', canvasWidth);
  canvas.setAttribute('height', canvasHeight);
  canvas.setAttribute('id', 'canvas');
  canvasDiv.appendChild(canvas);
  if(typeof G_vmlCanvasManager != 'undefined') {
    canvas = G_vmlCanvasManager.initElement(canvas);
  }
  context = canvas.getContext("2d");


  // Mouse Down Event
  $('#canvas').mousedown(function(e){
    var mouseX = e.pageX - this.offsetLeft;
    var mouseY = e.pageY - this.offsetTop;
          
    paint = true;
    addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
    redraw();
  });

  // Mouse Move Event
  $('#canvas').mousemove(function(e){
    if(paint){
      addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
      redraw();
    }
  });

  // Mouse Up Event
  $('#canvas').mouseup(function(e){
    paint = false;
  });

  // Mouse Leave Event
  $('#canvas').mouseleave(function(e){
    paint = false;
  });

  // addClick function to save the click position:
  var clickX = new Array();
  var clickY = new Array();
  var clickDrag = new Array();
  var paint;

  function addClick(x, y, dragging)
  {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
  }

  function redraw(){
    context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
    
    context.strokeStyle = "yellow";
    context.lineJoin = "round";
    context.lineWidth = 25;
              
    for(var i=0; i < clickX.length; i++) {        
      context.beginPath();
      if(clickDrag[i] && i){
        context.moveTo(clickX[i-1], clickY[i-1]);
       }else{
         context.moveTo(clickX[i]-1, clickY[i]);
       }
       context.lineTo(clickX[i], clickY[i]);
       context.closePath();
       context.stroke();
    }
  }
}

$("#predict").click(function(e) {
  // convert canvas to an image url
  var img = canvas.toDataURL();
  // make request to server
  $.post("predict/", {img: img, worked: "Yes"}, function(d) {
    // remove any pre-existing results
    // $("#results").children().remove()
    console.log(d);
    d3.select('#number').html(parseInt(d));

  });
  return false;
}); 


$(document).ready(function() {
  makeCanvas("canvasDiv");
  $("#clearCanvas").on("click", function (){
    d3.select("#canvas").remove()
    makeCanvas("canvasDiv")
  })
});