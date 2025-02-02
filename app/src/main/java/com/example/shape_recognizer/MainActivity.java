package com.example.shape_recognizer;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.example.model.DibujoView;
import com.example.shape_recognizer.databinding.ActivityMainBinding;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'shape_recognizer' library on application startup.
    static {
        System.loadLibrary("shape_recognizer");
    }

    private ActivityMainBinding binding;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        readCSVFromAssets(getAssets());
        readCSVZernikeFromAssets(getAssets());

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Configuración del Spinner (Combo Box)
        Spinner spinner = findViewById(R.id.spinnerOpciones);
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this,
                R.array.spinner_options, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);

        // Configuración del botón "Limpiar Lienzo"
        Button limpiarButton = binding.limpiarButton;
        TextView prediccionValor = findViewById(R.id.prediccionValor);
        limpiarButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Llamar al método limpiarLienzo de la vista DibujoView
                DibujoView dibujoView = binding.dibujoView;  // Acceder al DibujoView
                dibujoView.limpiarLienzo();  // Limpiar el lienzo
                prediccionValor.setText("Esperando prediccion... ");

            }
        });


        // Configuración del botón "Predecir"
        Button predecirButton = binding.predecirButton;
        predecirButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Obtener la opción seleccionada del Spinner
                String selectedOption = spinner.getSelectedItem().toString();

                DibujoView dibujoView = findViewById(R.id.dibujoView);
                Bitmap bitmap = dibujoView.getBitmap();

                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                try {
                    File file = new File(getExternalFilesDir(null), "debug_bitmap.png");
                    FileOutputStream fos = new FileOutputStream(file);
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
                    fos.close();
                    Log.i("Bitmap Debug", "Imagen guardada en: " + file.getAbsolutePath());
                } catch (IOException e) {
                    e.printStackTrace();
                }
                byte[] byteArray = stream.toByteArray();

                Log.i("Bitmap Info", "Width: " + bitmap.getWidth() + ", Height: " + bitmap.getHeight());

                if (selectedOption.equals("Hu")){
                    String predicted_class = huPrediction(byteArray);
                    prediccionValor.setText("Resultado de la predicción: " + predicted_class);
                } else if (selectedOption.equals("Zernike")){
                    String predicted_class = zernikePrediction(byteArray);
                    prediccionValor.setText("Resultado de la predicción: " + predicted_class);
                } else {
                    Toast.makeText(MainActivity.this, "Opcion Invalida", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    /**
     * A native method that is implemented by the 'shape_recognizer' native library,
     * which is packaged with this application.
     */
    public native void readCSVFromAssets(AssetManager assetManager);

    public native void readCSVZernikeFromAssets(AssetManager assetManager);

    public native String huPrediction(byte[] byteArray);

    public native String zernikePrediction(byte[] byteArray);
}
