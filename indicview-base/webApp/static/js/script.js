// hide buttons
$("#retry").hide()
$("#results").hide()

$(function() {

  // sanity check
  console.log("dom is ready!");

  $.ajaxSetup({ 
     beforeSend: function(xhr, settings) {
         function getCookie(name) {
             var cookieValue = null;
             if (document.cookie && document.cookie != '') {
                 var cookies = document.cookie.split(';');
                 for (var i = 0; i < cookies.length; i++) {
                     var cookie = jQuery.trim(cookies[i]);
                     // Does this cookie string begin with the name we want?
                     if (cookie.substring(0, name.length + 1) == (name + '=')) {
                         cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                         break;
                     }
                 }
             }
             return cookieValue;
         }
         if (!(/^http:.*/.test(settings.url) || /^https:.*/.test(settings.url))) {
             // Only send the token to relative URLs i.e. locally.
             xhr.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
         }
     } 
  });

  // event handler for form submission
  $('#post-form').on('submit', function(event){
    $("#results").hide()
    value = $('input[name="image_url"]').val();
    $.ajax({
      type: "POST",
      url: "/rango/ocr/",
      data : { "image_url" : value },
      success: function(result) {
        console.log(result);
        $("#post-form").hide()
        $("#retry").show()
        $("#results").show()
        $("#results").html("<h3>Image</h3><img src="+
          value+" style='max-width: 400px;'><br><h3>Results</h3><div class='well'>"+
          result["output"]+"</div>");
      },
      error: function(error) {
        console.log(error);
      }
    });
  });

  // Start search over, clear all existing inputs & results
  $('#retry').on('click', function(){
    $("input").val('').show();
    $("#post-form").show()
    $("#retry").hide()
    $('#results').html('');
  });


});