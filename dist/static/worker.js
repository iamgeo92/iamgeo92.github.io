// import * as Comlink from 'comlink'


// importScripts("https://unpkg.com/comlink/dist/umd/comlink.js");
// importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js");



importScripts("/static/scripts/comlink.js");
importScripts("/static/scripts/tf.min.js");


// new thing 
// importScripts("https://unpkg.com/@tensorflow-models/mobilenet");



const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
const IMAGE_SIZE = 224;
let pretrained_cnn;
let pretrained_feature_net ; 


let classification_model; // the model on top of the dense model which will be used 
let classification_tags ; 

let is_model_loaded = false; 


let feature_cache = {} ; // will be used during the training phase 


async function load_model( ){
	pretrained_cnn = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
	pretrained_feature_net =  tf.model({
		  inputs: pretrained_cnn.layers[0].input, 
		  outputs: [pretrained_cnn.layers[83].output , pretrained_cnn.layers[87].output]
		});

  await pretrained_feature_net.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]));

  // trying the new thing
  // pretrained_feature_net = await mobilenet.load();


	is_model_loaded = true; 
	// var imgElement = document.createElement('img');
	// console.log("model loased form worker")


	return true;
}





function Float32Concat(first, second)
{
    var firstLength = first.length,
        result = new Float32Array(firstLength + second.length);

    result.set(first);
    result.set(second, firstLength);

    return result;
}


function sleep (time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}






function load_virt_img(url){
  return new Promise((resolve, reject) => {
    const im = new Image()
        im.crossOrigin = 'anonymous'
        im.src = url
        im.onload = () => {
          resolve(im)
        }
   })
}

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}

// expect a stupid rgba img .. why RGBA coz fuk mah lyf 
async function get_feats( inp  , do_cache=true ){

  let img_buffer = inp.img_buffer;
  let img_id = inp.img_id; 

  if( do_cache )
  {
      if( img_id in feature_cache )
          return feature_cache[ img_id ];
  }
  

  let buff = new Float32Array(img_buffer.data);


  const logits = tf.tidy(() => {

    // var imgElement = document.createElement('img'); 
    // imgElement.src =  img_b64;

    // let img = tf.browser.fromPixels(imgElement).toFloat();
    let img = tf.tensor( buff  , [img_buffer.height , img_buffer.width , 4 ]);
    img = img.slice( [0,0,0], [img_buffer.height , img_buffer.width , 3 ] );

    // let img = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    img = tf.image.resizeBilinear( img, [IMAGE_SIZE, IMAGE_SIZE]); // 192,192 is dictated by my model

    const offset = tf.scalar(127.5);
    const normalized = img.sub(offset).div(offset);
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    let pre_feats = pretrained_feature_net.predict( batched );

    // new thing:
    // let pre_feats = pretrained_feature_net.infer(batched, 'conv_preds');

    return pre_feats;

  });

  let features = await logits[0].data();
  let features2 = await logits[1].data();

  // new thing 
  // let features = await logits.data();

  let ret =  Float32Concat( features2 , features );
  // let ret =  features;

  // console.log( "AMAXXX " + indexOfMax( features2 ));

  console.log("feeat form worker  ")
  console.log(ret )

  if( do_cache )
     feature_cache[ img_id ] = ret ; 

  return ret; 
}



async function full_predict(inp){
  let feats = await get_feats( inp , false);
  let pred_logi = await classification_model.predict( tf.tensor(([feats]) ));
  let pred_id = indexOfMax( await pred_logi.data() );
  console.log( " preddddd ");
  console.log( pred_logi);
  console.log( pred_id );
  return classification_tags[pred_id];

}



// input list of the batch_ids  and the classes 
async function train_batch( batch ){

  let batch_x = [];
  let batch_y = [];

  for( let i =0 ; i< batch.length ; i ++){
    batch_x.push(  feature_cache[batch[i].img ]  );
    let yy = new Array(classification_tags.length ).fill(0);
    yy[ classification_tags.indexOf( batch[i].label  ) ] = 1 ;
    batch_y.push( yy );

  }

  let x = tf.tensor(batch_x);
  let y = tf.tensor(batch_y);

  let ans = await  classification_model.trainOnBatch (x, y) ; // [ loss , acc ]
  console.log( ans );
  return ans;
}


async function init_model( tags  ){
   
   classification_tags = tags;

   let n_classes = tags.length;

   classification_model = tf.sequential({
     layers: [
       tf.layers.dense({inputShape: [1000 + 256  ], units: 32, activation: 'relu' , kernelInitializer: 'varianceScaling'}),
       tf.layers.dense({units: n_classes , activation: 'softmax' , kernelInitializer: 'varianceScaling' }),
     ]
   });



    classification_model.compile({
      optimizer: tf.train.adam() ,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    console.log("model initied ")

}

// so that the it can make sure that every feature is in the cache 
function if_feats_in_cache( id_list ){
  let ret = [];
  for( let idd of id_list ){
    if( !( idd in feature_cache ))
      return false; 
  }

  return true; 
}

// add items in the feature cache 
function set_feature_cache(){

}


function set_model_w(){

}

function get_model_w(){

}


const obj = {
  counter: 0,
  inc() {
    this.counter++;
  },
  load_model : load_model ,
  get_feats:get_feats ,
  init_model : init_model , 
  train_batch : train_batch , 
  full_predict : full_predict 

};





Comlink.expose(obj);


