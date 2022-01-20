


var COLORS ={
    'EMAIL':'#FDEE00',
    'ADDRESS':'#C32148',
    'PERSON':'#FE6F5E',
    'PHONENUMBER': '#9F8170',
    'MISCELLANEOUS':'#007BA7',
    'QUANTITY':'#D891EF',
    'PERSONTYPE':'#FF91AF',
    'ORGANIZATION':'#3DDC84',
    'PRODUCT':'#FBCEB1',
    'SKILL':'#B0BF1A',
    'IP':'#703642',
    'O': '#FFFFFF',
    'PAD': '#FFFFFF',
    'LOCATION':'#C0E8D5',
    'DATETIME':'aqua',
    'EVENT':'darkorange',
    'URL':'#BD33A4'
    }



// myFunction
function click_predict(interpret_show) {
    var text;

    if ($("textarea").val()!== "") {
        text = $("textarea").val();	
    }
    else{
        alert("please input text in textarea")
    }

    predict(text).done(function(r){
        var rs = r['rs'] 
        var clusted_results = rs
        console.log(rs)
        if (interpret_show == false){
            clusted_results = cluste_results(rs)
        }
        var html = conver_html(clusted_results, interpret_show)
        console.log(html)
        document.getElementById("rs-predict").innerHTML = html; 

    }
    );
    
}


function cluste_results(datas){
    console.log("cluste_results")
    console.log(datas)
    var clusted_results = []
    for (index in datas){
        lenght_data = clusted_results.length
        var word = datas[index][0]
        var tag_label = datas[index][1]
     

        if (lenght_data > 0 && tag_label == clusted_results[lenght_data - 1][1] && clusted_results[lenght_data - 1][0] != '[/n]' & word != '[/n]'){
            var temp = [clusted_results[lenght_data - 1][0] + " " + word, tag_label]
            clusted_results[lenght_data - 1] = temp
        }
        else{
            var temp = [word, tag_label]
            clusted_results.push(temp)

        } 
            
    }
    return clusted_results
}


function round(value, decimals) {
    return Number(Math.round(value+'e'+decimals)+'e-'+decimals);
}


function nomalize_score(value, list_value){
  
    var max = Math.max(...list_value)
    var min = Math.min(...list_value)

    console.log(min)
    console.log(max)

    
    if ((max - min) != 0){
        
        return (value - min)/(max - min)
    }

    else{
        return value/max
    }
}
function addRow(key,text, prob) {
    const div = document.createElement('div');
    div.className = 'row-2';
    div.id = 'row-2';
    div.innerHTML =  '<div class ="c-label"> <b>' + key + ": </b>" + (Math.round(prob * 100) / 100).toFixed(5)+'</div>' + '<div class="c-text">'+ text+'</div>';
    document.getElementById('rs-interpret').appendChild(div);
}
function generate_text(word, score, color){

    var word_generated = '<mark class="entity" style="background: '+color+'; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">' + word
        + '<span style="font-size: 0.6em; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">'+round(score, 2) +'</span>'

        + '</mark>'


    return word_generated

}


function control_view(){
    document.getElementById('loader_interpret').style.display = 'block'
    // console.log(index_word);
    const myNode = document.getElementById("rs-interpret");
    myNode.innerHTML = '';
}


function interpret(index_word) {
    
    control_view()

    var text; 
    if ($("textarea").val()!== "") {
        text = $("textarea").val();	
    }

    else{
        alert("please input text in textarea")
        return
    }

    api_interpret(index_word, text).done(function(r){

        if ("message" in r){
            var messsage = r["message"];
            alert(messsage)
            return
        }
        else{
            var sent = r["sent"];
            for (key in r){
            var text =""

            if (key != "sent"){
                var proba = r[key]["proba"];
                var scores = r[key]["scores"]
                var parts = sent.split(' ');
                var pos_list = []
                var neg_list = []
                for (index in  scores){
                    if (scores[index] >=0){
                        pos_list.push(scores[index])
                    }
                    else{
                        neg_list.push(-scores[index])
                    }
                }
               
                for (let i = 0; i < parts.length; i++) {
                    var word = parts[i];
                    var score = scores[i];
                    var color  = "rgba(128, 128, 128,0)"
                    if (score > 0){
                        var score_softmax = nomalize_score(score, pos_list);
                        color = "rgba(0,128,0,"+ score_softmax +")"
                    }
                    if (score < 0){
                        
                        var score_softmax = nomalize_score(-score, neg_list);
                        color = "rgba(255,0,0,"+ score_softmax +")"
                    }
                    text += generate_text(word,score,color);
                    
                }

                // turn off loader
                document.getElementById('loader_interpret').style.display = 'none'
                
                addRow(key,text, proba)
            }
        }

    }

    

    });

}




function conver_html(data, interpret_show){

    var content = ""
    var lenght_data = data.length
    console.log(data)
    for (index in data){
        var word = data[index][0]
        var tag = data[index][1]

        if (word.includes('/n')){
            content += word.replace('/n','</br>')
        }
        else if (tag == "O"){
            var text = '<a style="color: black">'+word+'</a>'
            content += text
        }

        else if (tag != "O"){
            var color_label = COLORS[tag]

            content += '<button style="background-color:' +color_label+'" class="btn btn-primary" type="button" id="dropdownMenuButton" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">'
            content += '<a style =" color: black;">'+word+'</a>'
            content +=  '<span style="color: black; font-size: 0.8em; font-weight: 900; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">'+tag+'</span>'
            
            if (interpret_show == true){
                content +=  '<i class="fas fa-info-circle"  style="margin: 2px; !important;" onclick=interpret('+index+')></i>'
            }

            content += '</button>' 
        }

        if (index !=  lenght_data -1){
            content += " "
        } 
            
    }

    return '<div class="entities" style="line-height: 2.5; direction: ltr">'+content+'</div>'

}

