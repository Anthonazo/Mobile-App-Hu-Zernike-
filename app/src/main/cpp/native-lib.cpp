#include <jni.h>
#include <string>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <android/log.h>

using namespace std;
using namespace cv;

// Estructura para almacenar la categoría y los momentos de Hu
struct HuMomentData {
    string category;             // Categoría de la forma
    vector<double> huMoments;   // Momentos de Hu (8 columnas)
};

struct ZernikeMomentData{
    string category;
    vector<double> zernikeMoments;
};

vector<HuMomentData> huMomentsData;

vector<ZernikeMomentData> zernikeMOmentsData;


// Función para leer y almacenar los datos del CSV
extern "C"
JNIEXPORT void JNICALL
Java_com_example_shape_1recognizer_MainActivity_readCSVFromAssets(JNIEnv *env, jobject thiz, jobject assetManager) {
    // Obtener el AssetManager desde el objeto Java
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    // Abrir el archivo CSV desde los assets
    AAsset* asset = AAssetManager_open(mgr, "hu_moments.csv", AASSET_MODE_UNKNOWN);

    if (asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "CSV Read", "No se pudo abrir el archivo CSV.");
        return;
    }

    // Leer el archivo completo en el buffer
    off_t file_length = AAsset_getLength(asset);
    char* buffer = new char[file_length];
    AAsset_read(asset, buffer, file_length);
    AAsset_close(asset);

    // Convertir el buffer a un string para poder procesarlo
    string content(buffer, file_length);
    delete[] buffer;

    // Procesar el contenido del archivo CSV
    stringstream ss(content);
    string line;

    // Saltarse la cabecera
    getline(ss, line);

    // Leer cada registro del CSV y almacenar en el vector global
    while (getline(ss, line)) {
        stringstream lineStream(line);
        string cell;
        HuMomentData row;  // Objeto de la estructura HuMomentData

        int column_count = 0;
        while (getline(lineStream, cell, ',')) {
            if (column_count == 0) {
                // La primera columna es la categoría
                row.category = cell;
            } else if (column_count < 8) {
                // Las siguientes columnas son los momentos de Hu
                row.huMoments.push_back(stod(cell)); // 'stod' convierte a double
            }
            column_count++;
        }

        // Almacenar la fila (registro) en el vector global
        huMomentsData.push_back(row);
    }

    // Mostrar el primer registro para verificar
    if (!huMomentsData.empty()) {
        __android_log_print(ANDROID_LOG_INFO, "CSV Read", "Primer registro guardado:");
        __android_log_print(ANDROID_LOG_INFO, "CSV Read", "Categoría: %s", huMomentsData[0].category.c_str());
        for (double moment : huMomentsData[0].huMoments) {
            __android_log_print(ANDROID_LOG_INFO, "CSV Read", "Celda: %f", moment);
        }
    }
}

// Función para leer y almacenar los datos del CSV
extern "C"
JNIEXPORT void JNICALL
Java_com_example_shape_1recognizer_MainActivity_readCSVZernikeFromAssets(JNIEnv *env, jobject thiz, jobject assetManager) {
    // Obtener el AssetManager desde el objeto Java
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    // Abrir el archivo CSV desde los assets
    AAsset* asset = AAssetManager_open(mgr, "zernike_moments.csv", AASSET_MODE_UNKNOWN);

    if (asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "CSV Read", "No se pudo abrir el archivo CSV.");
        return;
    }

    // Leer el archivo completo en el buffer
    off_t file_length = AAsset_getLength(asset);
    char* buffer = new char[file_length];
    AAsset_read(asset, buffer, file_length);
    AAsset_close(asset);

    // Convertir el buffer a un string para poder procesarlo
    string content(buffer, file_length);
    delete[] buffer;

    // Procesar el contenido del archivo CSV
    stringstream ss(content);
    string line;

    // Saltarse la cabecera
    getline(ss, line);

    // Leer cada registro del CSV y almacenar en el vector global
    while (getline(ss, line)) {
        stringstream lineStream(line);
        string cell;
        ZernikeMomentData row;  // Objeto de la estructura HuMomentData

        int column_count = 0;
        while (getline(lineStream, cell, ',')) {
            if (column_count == 0) {
                // La primera columna es la categoría
                row.category = cell;
            } else if (column_count < 26) {
                // Las siguientes columnas son los momentos de Hu
                row.zernikeMoments.push_back(stod(cell)); // 'stod' convierte a double
            }
            column_count++;
        }

        // Almacenar la fila (registro) en el vector global
        zernikeMOmentsData.push_back(row);
    }

    // Mostrar el primer registro para verificar
    if (!zernikeMOmentsData.empty()) {
        __android_log_print(ANDROID_LOG_INFO, "CSV Read", "Primer registro guardado:");
        __android_log_print(ANDROID_LOG_INFO, "CSV Read", "Categoría: %s", zernikeMOmentsData[0].category.c_str());
        for (double moment : zernikeMOmentsData[0].zernikeMoments) {
            __android_log_print(ANDROID_LOG_INFO, "CSV Read", "Celda: %f", moment);
        }
    }
}

// Función para calcular la distancia Euclidiana entre dos registros (filas) usando los momentos de Hu
double calculateEuclideanDistance(const HuMomentData& row1, const HuMomentData& row2) {
    if (row1.huMoments.size() != row2.huMoments.size()) {
        return -1;  // Retorna un valor negativo si las filas no son del mismo tamaño
    }
    double sum = 0;
    for (size_t i = 0; i < row1.huMoments.size(); ++i) {
        sum += pow(row1.huMoments[i] - row2.huMoments[i], 2);  // Sumar las diferencias al cuadrado
    }
    return sqrt(sum);  // Retorna la raíz cuadrada de la suma
}

double calculateEuclideanDistanceZernike(const ZernikeMomentData& row3, const ZernikeMomentData& row4) {
    if (row3.zernikeMoments.size() != row4.zernikeMoments.size()) {
        __android_log_print(ANDROID_LOG_INFO, "RO1", "Celda: %i", row3.zernikeMoments.size());
        __android_log_print(ANDROID_LOG_INFO, "RO2", "Celda: %i", row4.zernikeMoments.size());
        return -1;  // Retorna un valor negativo si las filas no son del mismo tamaño
    }
    double sum = 0;
    for (size_t i = 0; i < row3.zernikeMoments.size(); ++i) {
        sum += pow(row3.zernikeMoments[i] - row4.zernikeMoments[i], 2);  // Sumar las diferencias al cuadrado
    }
    return sqrt(sum);  // Retorna la raíz cuadrada de la suma
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_shape_1recognizer_MainActivity_huPrediction(JNIEnv *env, jobject thiz, jbyteArray byte_array) {
    jbyte *data = env->GetByteArrayElements(byte_array, NULL);
    jsize length = env->GetArrayLength(byte_array);

    // Decodificar la imagen desde el arreglo de bytes
    cv::Mat image = cv::imdecode(cv::Mat(1, length, CV_8U, data), cv::IMREAD_GRAYSCALE);

    // Liberar los recursos del arreglo de bytes
    env->ReleaseByteArrayElements(byte_array, data, 0);

    if (image.empty()) {
        return env->NewStringUTF("Error al cargar la imagen.");
    }

    // Binarizar la imagen (invertir la escala de grises)
    cv::Mat binary_image;
    cv::threshold(image, binary_image, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    // Encontrar el bounding box del objeto
    cv::Rect bbox = cv::boundingRect(binary_image);  // Obtener el bounding box del objeto

    // Verificar si el bounding box tiene un área válida
    if (bbox.width == 0 || bbox.height == 0) {
        return env->NewStringUTF("Bounding box vacío, no se detectó objeto.");
    }

    // Recortar la región de interés (ROI) usando el bounding box
    cv::Mat roi = binary_image(bbox);  // Recortar la imagen utilizando el bounding box

    // Calcular los momentos de la imagen binarizada recortada (ROI)
    cv::Moments moments = cv::moments(roi, true);  // Momentos centrados

    // Calcular los momentos de Hu
    double huMoments[7];
    cv::HuMoments(moments, huMoments);  // Calcular momentos de Hu

    // Aplicar la transformación logarítmica a los momentos de Hu
    double huMomentsLog[7];
    for (int i = 0; i < 7; i++) {
        if (huMoments[i] != 0) {
            huMomentsLog[i] = -std::copysign(1.0, huMoments[i]) * std::log10(std::abs(huMoments[i]) + 1e-10);  // Evitar log(0)
        } else {
            huMomentsLog[i] = 0;  // Si el momento es 0, se establece 0 para evitar log(0)
        }
    }

    // Crear un HuMomentData para la imagen entrante
    HuMomentData incomingData;
    incomingData.category = "unknown";  // Aquí no se sabe la categoría, es solo para el cálculo
    for (int i = 0; i < 7; ++i) {
        incomingData.huMoments.push_back(huMomentsLog[i]);
    }

    // Comparar con los registros cargados del CSV
    double minDistance = std::numeric_limits<double>::infinity();
    string predictedCategory = "Desconocido";  // Valor por defecto si no hay coincidencias

    // Recorrer todos los registros cargados desde el CSV
    for (const HuMomentData& row : huMomentsData) {
        // Calcular la distancia Euclidiana con los momentos de Hu de la imagen entrante
        double distance = calculateEuclideanDistance(row, incomingData);

        // Si la distancia es más pequeña que la mínima, actualizar la categoría
        if (distance < minDistance) {
            minDistance = distance;
            predictedCategory = row.category;  // Asignar la categoría correspondiente

            // Mostrar los momentos de Hu más cercanos y la distancia
            __android_log_print(ANDROID_LOG_INFO, "Predicción", "Valor Entrante de Hu Moments:");
            for (int i = 0; i < 7; i++) {
                __android_log_print(ANDROID_LOG_INFO, "Predicción", "Momento %d: %f", i+1, huMomentsLog[i]);
            }

            __android_log_print(ANDROID_LOG_INFO, "Predicción", "Valor Cercano de Hu Moments (Categoría: %s):", row.category.c_str());
            for (int i = 0; i < 7; i++) {
                __android_log_print(ANDROID_LOG_INFO, "Predicción", "Momento %d: %f", i+1, row.huMoments[i]);
            }

            __android_log_print(ANDROID_LOG_INFO, "Predicción", "Distancia Euclidiana: %f", distance);
        }
    }

    // Mostrar la categoría predicha
    __android_log_print(ANDROID_LOG_INFO, "Predicción Final", "Categoría Predicha: %s", predictedCategory.c_str());

    // Retornar la categoría predicha
    return env->NewStringUTF((predictedCategory).c_str());
}

double factorial(int n) {
    double result = 1.0;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Zernike Polynomial calculation
double radialPolynomial(int n, int m, double r) {
    double sum = 0.0;
    for (int k = 0; k <= (n - abs(m)) / 2; k++) {
        double num = pow(-1, k) * factorial(n - k);
        double denom = factorial(k) * factorial((n + abs(m)) / 2 - k) * factorial((n - abs(m)) / 2 - k);
        sum += (num / denom) * pow(r, n - 2 * k);
    }
    return sum;
}

// Compute Zernike moments
vector<double> computeZernikeMomentss(Mat& img, int radius, int order) {
    int cx = img.cols / 2;
    int cy = img.rows / 2;
    vector<double> moments;

    for (int n = 0; n <= order; n++) {
        for (int m = -n; m <= n; m += 2) {
            complex<double> moment(0.0, 0.0);

            for (int y = 0; y < img.rows; y++) {
                for (int x = 0; x < img.cols; x++) {
                    double dx = (x - cx) / (double)radius;
                    double dy = (y - cy) / (double)radius;
                    double r = sqrt(dx * dx + dy * dy);
                    double theta = atan2(dy, dx);

                    if (r <= 1.0) {
                        double R = radialPolynomial(n, m, r);
                        complex<double> Z = R * exp(complex<double>(0, -m * theta));
                        moment += (double)img.at<uchar>(y, x) * Z;
                    }
                }
            }
            moments.push_back(abs(moment));
        }
    }
    return moments;
}


// Factorial function using memoization
double fact(int n) {
    static vector<double> factorial_table = {1, 1};
    while ((int)factorial_table.size() <= n) {
        factorial_table.push_back(factorial_table.back() * factorial_table.size());
    }
    return factorial_table[n];
}

// Compute the center of mass of the image
Point2d center_of_mass(const Mat& img) {
    Moments m = moments(img, true);
    return Point2d(m.m10 / m.m00, m.m01 / m.m00);
}

// Compute the Zernike moment for given (n, l)
complex<double> znl(const vector<double>& Dn, const vector<complex<double>>& An, const vector<double>& P, int n, int l) {
    const double pi = atan(1.0) * 4;
    vector<double> g_m((n - l) / 2 + 1);

    for (int m = 0; m <= (n - l) / 2; m++) {
        double f = (m & 1) ? -1 : 1;
        g_m[m] = f * fact(n - m) / (fact(m) * fact((n - 2 * m + l) / 2) * fact((n - 2 * m - l) / 2));
    }

    complex<double> Vnl = 0.0;
    for (size_t i = 0; i < Dn.size(); ++i) {
        complex<double> sum_term = 0.0;
        for (int m = 0; m <= (n - l) / 2; m++) {
            sum_term += g_m[m] * pow(Dn[i], n - 2 * m) * An[i];
        }
        Vnl += P[i] * conj(sum_term);
    }

    return Vnl * ((n + 1) / pi);
}

// Meshgrid function in C++
void meshgrid(const Range& x_range, const Range& y_range, Mat& X, Mat& Y) {
    vector<double> x_values, y_values;
    for (int i = x_range.start; i < x_range.end; ++i) x_values.push_back(i);
    for (int j = y_range.start; j < y_range.end; ++j) y_values.push_back(j);

    X = Mat(y_values.size(), x_values.size(), CV_64F);
    Y = Mat(y_values.size(), x_values.size(), CV_64F);

    for (size_t i = 0; i < y_values.size(); ++i) {
        for (size_t j = 0; j < x_values.size(); ++j) {
            X.at<double>(i, j) = x_values[j];
            Y.at<double>(i, j) = y_values[i];
        }
    }
}

// Compute Zernike moments for an image
vector<double> computeZernikeMoments(const Mat& img, int radius, int degree) {
    Point2d cm = center_of_mass(img);
    int rows = img.rows, cols = img.cols;
    vector<double> zvalues;

    Mat Y, X;
    meshgrid(Range(0, rows), Range(0, cols), X, Y);

    Y.convertTo(Y, CV_64F);
    X.convertTo(X, CV_64F);
    Mat P;
    img.convertTo(P, CV_64F);
    P /= sum(P)[0];

    Mat Yn = (Y - cm.y) / radius;
    Mat Xn = (X - cm.x) / radius;
    Mat Dn;
    magnitude(Xn, Yn, Dn);
// Aseguramos que todas las matrices tengan el mismo tamaño
    Dn = Dn.reshape(1, img.rows);
    P = P.reshape(1, img.rows);

// Asegurar que las matrices sean del mismo tipo antes de la comparación
    Mat mask = (Dn <= 1.0) & (P > 0.0);
    mask.convertTo(mask, CV_8U);



    vector<double> Dn_vec, P_vec;
    vector<complex<double>> An_vec;
    for (int i = 0; i < mask.total(); i++) {
        if (mask.at<uchar>(i)) {
            double dx = Xn.at<double>(i);
            double dy = Yn.at<double>(i);
            double d = Dn.at<double>(i);
            complex<double> a(dx / d, dy / d);
            Dn_vec.push_back(d);
            An_vec.push_back(a);
            P_vec.push_back(P.at<double>(i));
        }
    }

    for (int n = 0; n <= degree; n++) {
        for (int l = 0; l <= n; l++) {
            if ((n - l) % 2 == 0) {
                complex<double> z = znl(Dn_vec, An_vec, P_vec, n, l);
                zvalues.push_back(abs(z));
            }
        }
    }
    return zvalues;
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_shape_1recognizer_MainActivity_zernikePrediction(JNIEnv *env, jobject thiz, jbyteArray byte_array) {
    jbyte *data = env->GetByteArrayElements(byte_array, NULL);
    jsize length = env->GetArrayLength(byte_array);

    cv::Mat image = cv::imdecode(cv::Mat(1, length, CV_8U, data), cv::IMREAD_GRAYSCALE);

    // Liberar los recursos del arreglo de bytes
    env->ReleaseByteArrayElements(byte_array, data, 0);

    if (image.empty()) {
        return env->NewStringUTF("Error al cargar la imagen.");
    }

    // Binarizar la imagen (invertir la escala de grises)
    cv::Mat binary_image;
    cv::threshold(image, binary_image, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    // Encontrar el bounding box del objeto
    cv::Rect bbox = cv::boundingRect(binary_image);  // Obtener el bounding box del objeto

    // Verificar si el bounding box tiene un área válida
    if (bbox.width == 0 || bbox.height == 0) {
        return env->NewStringUTF("Bounding box vacío, no se detectó objeto.");
    }

    // Recortar la región de interés (ROI) usando el bounding box
    cv::Mat roi = binary_image(bbox);  // Recortar la imagen utilizando el bounding box

    int radius = min(roi.rows, roi.cols) / 2;

    ZernikeMomentData incomingData;
    incomingData.zernikeMoments = computeZernikeMoments(roi, radius, 8);
    incomingData.category = "unknown";  // Aquí no se sabe la categoría, es solo para el cálculo

    double minDistance = std::numeric_limits<double>::infinity();
    string predictedCategory = "Desconocido";  // Valor por defecto si no hay coincidencias

    // Recorrer todos los registros cargados desde el CSV
    for (const ZernikeMomentData& row : zernikeMOmentsData) {
        // Calcular la distancia Euclidiana con los momentos de Hu de la imagen entrante
        double distance = calculateEuclideanDistanceZernike(row, incomingData);

        // Si la distancia es más pequeña que la mínima, actualizar la categoría
        if (distance < minDistance) {
            minDistance = distance;
            predictedCategory = row.category;  // Asignar la categoría correspondiente

            // Mostrar los momentos de Hu más cercanos y la distancia
            __android_log_print(ANDROID_LOG_INFO, "Predicción", "Distancia Euclidiana: %f", distance);
            __android_log_print(ANDROID_LOG_INFO, "Predicción", "Valor Cercano de Zernike (Categoría: %s):", row.category.c_str());

        }
    }

    return env->NewStringUTF(predictedCategory.c_str());
}



