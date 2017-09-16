
//Constituir objetos 
  const preObject = document.getElementById("pre");
  const ulList = document.getElementById("list");
  const divResults = document.getElementById("events");

//Crear una referencia con firebase
  const dbRefObject = firebase.database().ref().child('object');
  const dbRefList = dbRefObject.child('hobbies');
  const dbRefEvents = dbRefObject.child('events');

//Encender la conexión y comunicar un snapshot 
//dbRefObject.on('value', snap => console.log(snap.val()));
//Mostrar con formato JSON el resultado de la base.
  dbRefObject.on('value', snap => {
    divResults.innerText = JSON.stringify(snap.val(), null, 3);
  });


//vista de los datos subidos con Google Analytics
  dbRefEvents.on('value', snap => {
    divResults.innerText = JSON.stringify(snap.val(), null, 3);
  });

//modificar vista con base de datos
  dbRefList.on('child_added', snap => {
    const li = document.createElement('li');
    li.innerText = snap.val();
    li.id = snap.key;
    ulList.appendChild(li);
  });
 
//mofificar vista en un elemento li si cambia la bdd
dbRefList.on('child_changed', snap => {
  const liChanged = document.getElementById(snap.key);
  liChanged.innerText = snap.val();
});


//remover si se removio un elemento
dbRefList.on('child_removed', snap => {
  const liToRemove = document.getElementById(snap.key);
  liToRemove.remove();
});
