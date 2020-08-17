
  // Your web app's Firebase configuration
  var firebaseConfig = {
    apiKey: "AIzaSyDZK1ZxypcbM_eoFjY7snuo5_MtjmOzgf0",
    authDomain: "key-stroke-dynamics-c823e.firebaseapp.com",
    databaseURL: "https://key-stroke-dynamics-c823e.firebaseio.com",
    projectId: "key-stroke-dynamics-c823e",
    storageBucket: "key-stroke-dynamics-c823e.appspot.com",
    messagingSenderId: "302519402835",
    appId: "1:302519402835:web:695f9b9c9ca48e4a3de1a6",
    measurementId: "G-J8SR7HC2RJ"
  };
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
  var messagesRef = firebase.database().ref('key_stroke_data');

  document.addEventListener('DOMContentLoaded', init);
  let log = console.log;


  let key_stroke_timings = {}

  function init(){
      let txt = document.getElementById('txt');
      txt.addEventListener('keydown', keydown_handler);
      txt.addEventListener("keyup",keyup_handler);

      let ex_btn = document.getElementById("export_btn");
      ex_btn.addEventListener("click",saveMessage_toFirebase);

  }

  function keydown_handler(ev){
      //let temp = ev.timeStamp;
      //log("temp: ", temp)
      let key_press_timestamp = new Date();
      let char = ev.char || ev.charCode || ev.which;
      let s = String.fromCharCode(char);
      log("You entered: ", s," at time: ", key_press_timestamp.getTime());
      add_value(key_stroke_timings,"key_pressed_timestamp",key_press_timestamp.getTime())
      add_value(key_stroke_timings,"key_pressed",s)
  }

  function keyup_handler(ev){
      let key_release_time_stamp = new Date();
      let char = ev.char || ev.charCode || ev.which;
      let s = String.fromCharCode(char);
      log("You released: ", s," at time: ", key_release_time_stamp.getTime());
      add_value(key_stroke_timings,"key_released_timestamp",key_release_time_stamp.getTime())
      add_value(key_stroke_timings,"key_released",s)
      log(key_stroke_timings)
  }

  function add_value(obj,key,value){
    if(obj.hasOwnProperty(key)){
      obj[key].push(value);
    }else{
      obj[key] = [value];
    }
  }

  function show_curr_data(ev){
    alert(key_stroke_timings);
  }
  function saveMessage_toFirebase(ev){
    let curr_user = getInputValue("user_name")
    let sess_num = getInputValue("session_num")
    add_value(key_stroke_timings,"username",curr_user)
    add_value(key_stroke_timings,"session_num",sess_num)
    var newMessageRef = messagesRef.push();
    let jsonString = JSON.stringify(key_stroke_timings);
    newMessageRef.set(jsonString)
    alert("Keystroke data submitted to firebase database!!")
}

  function getInputValue(id){
    return document.getElementById(id).value;
  }
