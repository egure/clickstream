(function(){
  // Initialize Firebase
  var config = {
    apiKey: "AIzaSyDUkZoY9nE97BRl1eGDWGuTdfnJX9X960c",
    authDomain: "neem-fs.firebaseapp.com",
    databaseURL: "https://neem-fs.firebaseio.com",
    projectId: "neem-fs",
    storageBucket: "neem-fs.appspot.com",
    messagingSenderId: "455316581334"
  };
  firebase.initializeApp(config);

//Sync object 
  const preObject = document.getElementById('object');

//Sync object 
  const dbRefObject = firebase.database().ref().child('object');

  dbRefObject.on('value', snap => console.log(snap.val()));

//Sync object 
  dbRefObject.on('value', snap => {
    preObject.innerText = JSON.stringify(snap.val(), null, 3);
  });

}());