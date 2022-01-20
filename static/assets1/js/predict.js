
var URL_ROOT = ''

function predict(text) {
    var content_dict = {"text": text}
    var path = URL_ROOT + "predict"
    return ajax_api(content_dict,path,  type="POST")
    
}


function api_interpret(index_word, text) {
    var content_dict = { "text": text, "idx": index_word}
    var path = URL_ROOT + "interpret"
    return ajax_api(content_dict,path,  type="POST")

}


function ajax_api(content_dict, path, type){
    return $.ajax({
        url: path,
        data: 
            JSON.stringify(
                content_dict
            ),
        type: type,
        contentType: "application/json; charset=utf-8",
    });
}


