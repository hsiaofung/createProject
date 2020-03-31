var manager = new Mananger();
manager.init("container_0", 500, 375);
manager.initShadow("container_1", 500, 375);

var count = 0;
var selectAll = 0;
function glLinkProgramCallback(t) {
  if (t == 0) {
    $("#loadingDiv").show();
  } else {
    $("#loadingDiv").hide();
  }
}
function render() {
  manager.render();
}
var isUpdated = 0;
function animate() {
  requestAnimationFrame(animate);
  if (isUpdated > 0) {
    // $(".logoEdit").change();
    changeLogo();
    isUpdated -= 1;
  }
  render();
}

animate();

function selectRing(id) {
  console.log(manager);
  if (id == 2) {
    selectAll = 1;
  } else {
    selectAll = 0;
    manager.selected_id = id;
    var ring = manager.selectedRing();
    $("#shape_select").val(ring.shapeType);
    $("#width_select").val(ring.widthType);
    $("#gold_select").val(ring.colorType);
    $("#show_out_select").val(ring.isShowOutterDiamond);
    $("#show_in_select").val(ring.isShowInnerDiamond);
    $("#surface_select").val(ring.surfaceType);
  }
}
$("input[name=sex]").get(2).checked = true;

selectRing(0);
selectRing(2);

$("#shape_select").change(function() {
  manager.changeShapeType($(this).val());
  if (selectAll) {
    manager.selected_id = (manager.selected_id + 1) % 2;
    manager.changeShapeType($(this).val());
  }
});

$("#width_select").change(function() {
  manager.changeWidthType($(this).val());
  if (selectAll) {
    manager.selected_id = (manager.selected_id + 1) % 2;
    manager.changeWidthType($(this).val());
  }
});

$("#gold_select").change(function() {
  manager.changeColorType($(this).val());
  if (selectAll) {
    manager.selected_id = (manager.selected_id + 1) % 2;
    manager.changeColorType($(this).val());
  }
});

$("#show_out_select").change(function() {
  manager.selectedRing().showOutterDiamond($(this).val());
  if (selectAll) {
    manager.selected_id = (manager.selected_id + 1) % 2;
    manager.selectedRing().showOutterDiamond($(this).val());
  }
});

$("#show_in_select").change(function() {
  manager.selectedRing().showInnerDiamond($(this).val());
  if (selectAll) {
    manager.selected_id = (manager.selected_id + 1) % 2;
    manager.selectedRing().showInnerDiamond($(this).val());
  }
});

$("#surface_select").change(function() {
  manager.selectedRing().changeSurfaceType($(this).val());
  if (selectAll) {
    manager.selected_id = (manager.selected_id + 1) % 2;
    manager.selectedRing().changeSurfaceType($(this).val());
  }
});

$("#in_select_color").change(function() {
  manager.selectedRing().changeInnerDiamondColor($(this).val());
  if (selectAll) {
    manager.selected_id = (manager.selected_id + 1) % 2;
    manager.selectedRing().changeInnerDiamondColor($(this).val());
  }
});

$("#out_select_color").change(function() {
  manager.selectedRing().changeOutterDiamondColor($(this).val());
  if (selectAll) {
    manager.selected_id = (manager.selected_id + 1) % 2;
    manager.selectedRing().changeOutterDiamondColor($(this).val());
  }
});

$("#ringnum_select").change(function() {
  manager.showRingMode($(this).val());
  if ($(this).val() == 1) {
    selectRing(0);
    $("input[name=sex]").get(0).checked = true;
  }
});

function changeLogo() {
  var sc = parseFloat($("#logoSc").val());
  var ox = parseFloat($("#logoOx").val());
  var oy = parseFloat($("#logoOy").val());
  var left = parseFloat($("#logoLeft").val());

  manager.changeLogoPos(sc, ox, oy);
  if (selectAll) {
    manager.selected_id = (manager.selected_id + 1) % 2;
    manager.changeLogoPos(sc, ox, oy);
  }
}

$(".logoEdit").change(function() {
  // console.log('edit logo');
  changeLogo();
});

$("#logoSc").val(1.25);
changeLogo();
isUpdated = 4;

$("#gold_select").change();
