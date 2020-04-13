package apps.nocturnuslabs.sillymusicplayer;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class ClassifyActivity extends AppCompatActivity {

    // presets for rgb conversion
    private static final int RESULTS_TO_SHOW = 3;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    private Interpreter tflite; //tflite graph
    private List<String>labelList; //possible labels for the model
    private ByteBuffer imgData = null; //selected image as bytes
    private byte[][] labelProbArray = null; //probabilities for each label
    private String [] topLables = null; //labels with highest probabilities
    private String[] topConfidence = null; //array with highest probabilities

    // input image dimensions for the Model
    private int DIM_IMG_SIZE_X = 224;
    private int DIM_IMG_SIZE_Y = 224;
    private int DIM_PIXEL_SIZE = 3;

    // int array to hold image data
    private int[] intValues;

    // priority queue that will hold the top results from the CNN
    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    private ImageView selectedImage;
    private TextView moodtxt;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        // initialize array that holds image data
        intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

        super.onCreate(savedInstanceState);

        try{
            tflite = new Interpreter(loadModelFile(), tfliteOptions);
            labelList = loadLabelList();
        } catch (Exception ex){
            ex.printStackTrace();
        }

        //initialize byte array based on the model
        imgData = ByteBuffer.allocateDirect(DIM_IMG_SIZE_X*DIM_IMG_SIZE_Y*DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());

        //initialize probabilities array
        labelProbArray = new byte[1][labelList.size()];

        setContentView(R.layout.activity_classify);
        selectedImage = findViewById(R.id.classify_imageview);
        moodtxt = findViewById(R.id.classify_moodtxt);

        topLables = new String[RESULTS_TO_SHOW];
        topConfidence = new String[RESULTS_TO_SHOW];

        Uri uri = getIntent().getParcelableExtra("Image_URI");
        try{
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(),uri);
            selectedImage.setImageBitmap(bitmap);
        }catch (IOException e) {
            e.printStackTrace();
        }

        classifyImage();
    }

    private void classifyImage() {
        Bitmap bitmap_orignal = ((BitmapDrawable)selectedImage.getDrawable()).getBitmap();
        Bitmap bitmap = getResizedBitmap(bitmap_orignal,DIM_IMG_SIZE_X,DIM_IMG_SIZE_Y);
        convertBitmaptoByteBuffer(bitmap);

        tflite.run(imgData,labelProbArray);
        printTopKLabels();
    }

    //print labels and their corresponding confidence
    private void printTopKLabels() {
        for(int i =0;i<labelList.size();++i){ //add all results in priority queue
            sortedLabels.add(new AbstractMap.SimpleEntry<String, Float>(labelList.get(i),(labelProbArray[0][i]&0xff)/255.0f));
            if(sortedLabels.size()>RESULTS_TO_SHOW){
                sortedLabels.poll();
            }
        }
        //get top results from priority queue
        final int size = sortedLabels.size();
        for(int i = 0; i < size; ++i){
            Map.Entry<String,Float> label = sortedLabels.poll();
            assert label != null;
            topLables[i] = label.getKey();
            topConfidence[i] = String.format("%.0f%%",label.getValue()*100);
        }

        moodtxt.setText("You seem "+topLables[1]+" with probability of "+topConfidence[1]);

    }

    private void convertBitmaptoByteBuffer(Bitmap bitmap) {
        if(imgData==null){
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues,0, bitmap.getWidth(),0,0,bitmap.getWidth(), bitmap.getHeight());
        //loop through all pixels
        int pixel=0;
        for(int i = 0; i<DIM_IMG_SIZE_X;++i){
            for(int j = 0; j<DIM_IMG_SIZE_Y;++j){
                final int val = intValues[pixel++];
                imgData.put((byte)((val>>16)&0xFF));
                imgData.put((byte)((val>>8)&0xFF));
                imgData.put((byte)((val)&0xFF));
            }
        }
    }

    //resizes bitmap to required dimensions
    public Bitmap getResizedBitmap(Bitmap bitmap, int dim_x, int dim_y) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float scaleWidth = ((float)dim_x)/width;
        float scaleHeight = ((float)dim_y)/height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth,scaleHeight);
        return Bitmap.createBitmap(bitmap,0,0,width,height,matrix,false);
    }

    //Get the labels of the model in the labelList
    private List<String> loadLabelList() throws IOException{
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(this.getAssets().open("labels.txt")));
        String line;
        while((line=reader.readLine())!=null){
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    //Get the tflite model in form of a ByteBuffer
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("emotions.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
}

