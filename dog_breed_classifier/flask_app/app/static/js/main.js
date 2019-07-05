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

    // store the image temporarily, so we can use it
    var img = event.originalEvent.dataTransfer.files[0];
    USER_IMAGE = img;
    // update the input field as well for consistency
    //$("#img_input").val(img.name);
    document.querySelector('#img_input').files = event.originalEvent.dataTransfer.files;
    // render the image as thumbnail
    renderImage(img);
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
        $('#img_thumbnail').html('<img class="img-fluid fixed-size rounded" src="' + the_url + '"  alt="Some image"/>');
    }

    // when the file is read it triggers the onload event above.
    reader.readAsDataURL(file);
}

function displayClassification(response){
    console.log('Success');
    console.log(response);
    $('#responseDiv').html(response.message)
}

function displayError(qXHR, textStatus, errorMessage){
    console.log(errorMessage);
    $('#responseDiv').html('<p  class="result">Woops! En error occurred. Oh noes. Not sure what happened. Can you try it again?</p>')
}

function requestClassification() {
    // show loader
    $('#responseDiv').html('<div class="result"><img class="img-fluid" src="static/images/ripple-1.5s-200px.svg" alt="Loading ..."></div>')

    // load image into FormData
    var formData = new FormData();
    formData.append("img", USER_IMAGE);

    console.log(formData);

    // transfer image to server
    $.ajax({
        url: '/classify',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: displayClassification,
        error:  displayError
      });
}

function updateNav() {
    $('#navigation').find('li').removeClass('active');
    $('#navigation').find('a[href="'+ navActive +'"]').addClass('active');
}

var main = function () {
    //console.log('Hello World!');
    updateNav();
    if (navActive == '/') {
        $("#img_input").change(fileInput);
        $('#img_card').on('dragover', handleDragOver);
        $('#img_card').on('dragenter', handleDragEnter);
        $('#img_card').on('drop', fileDrop);
        $('#classify_button').click(requestClassification)
    }
}
