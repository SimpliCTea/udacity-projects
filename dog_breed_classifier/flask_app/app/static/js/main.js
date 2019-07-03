var USER_IMAGE = null;


function fileInput(event){
        // will log a FileList object, view gifs below
        console.log('File selected.');
        console.log(event);
        console.log(event.target.files);
        USER_IMAGE = event.target.files[0];
        renderImage(USER_IMAGE);
}

function fileDrop(event){
    event.stopPropagation();
    event.preventDefault();

    console.log(event)

    var file = event.originalEvent.dataTransfer.files[0];

    renderImage(file);
}

function handleDragOver(event) {
    console.log(event);
    event.stopPropagation();
    event.preventDefault();
    //event.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
}

function handleDragEnter(event) {
    console.log(event);
    event.stopPropagation();
    event.preventDefault();
    //event.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
}

function renderImage(file){
    // generate a new FileReader object
    var reader = new FileReader();

    // inject an image with the src url
    reader.onload = function(event) {
        the_url = event.target.result;
        $('#img_holder').html('<img class="img-fluid" src="' + the_url + '" />');
    }

    // when the file is read it triggers the onload event above.
    reader.readAsDataURL(file);
}

var main = function () {
    console.log('Hello World!');
    $("#img_input").change(fileInput);
    $('#img_holder').on('dragover', handleDragOver);
    $('#img_holder').on('dragenter', handleDragEnter);
    $('#img_holder').on('drop', fileDrop);
}
