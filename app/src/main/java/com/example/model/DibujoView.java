package com.example.model;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class DibujoView extends View {
    private Paint paint;
    private Path path;
    private Bitmap bitmap;
    private Canvas canvas;

    // Constructor que acepta un contexto
    public DibujoView(Context context) {
        super(context);
        init();
    }

    // Constructor que acepta un contexto y un conjunto de atributos
    public DibujoView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        paint = new Paint();
        paint.setColor(Color.BLACK);  // Color del trazo
        paint.setAntiAlias(true);     // Suavizar los bordes
        paint.setStrokeWidth(5f);     // Grosor del trazo
        paint.setStyle(Paint.Style.STROKE);  // Solo contorno, no relleno
        path = new Path();
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        // Crear un Bitmap para pintar en él
        bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        canvas = new Canvas(bitmap);
        canvas.drawColor(Color.WHITE);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        // Dibujar sobre el canvas original (pantalla)
        canvas.drawBitmap(bitmap, 0, 0, null);
        canvas.drawPath(path, paint); // Dibujar el trazo actual
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y);
                return true;
            case MotionEvent.ACTION_MOVE:
                path.lineTo(x, y);
                break;
            case MotionEvent.ACTION_UP:
                canvas.drawPath(path, paint);
                path.reset();
                break;
        }
        invalidate();  // Redibuja la vista
        return true;
    }

    // Método para limpiar el lienzo
    public void limpiarLienzo() {
        bitmap.eraseColor(Color.WHITE);  // Rellenar el bitmap con blanco (limpiar)
        invalidate();  // Redibuja la vista para que se vea el lienzo limpio
    }

    public Bitmap getBitmap() {
        return bitmap;
    }

}