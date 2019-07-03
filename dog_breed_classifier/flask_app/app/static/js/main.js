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

    var file = event.originalEvent.dataTransfer.files[0];
    USER_IMAGE = file;

    renderImage(file);
}

function handleDragOver(event) {
    event.stopPropagation();
    event.preventDefault();
    //event.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
}

function handleDragEnter(event) {
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

function displayClassification(response){
    console.log('Success');
    console.log(response);
}

function requestClassification() {
    //$.get('/classify', file, displayClassification);
    var formData = new FormData();
    formData.append("img", USER_IMAGE);

    console.log(formData);

    $.ajax({
        url: '/classify',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: displayClassification,
        error:  function(jqXHR, textStatus, errorMessage) {
            console.log(errorMessage); // Optional
        }
      });
}

var main = function () {
    console.log('Hello World!');
    $("#img_input").change(fileInput);
    $('#img_holder').on('dragover', handleDragOver);
    $('#img_holder').on('dragenter', handleDragEnter);
    $('#img_holder').on('drop', fileDrop);
    $('#classify_button').click(requestClassification)
}
