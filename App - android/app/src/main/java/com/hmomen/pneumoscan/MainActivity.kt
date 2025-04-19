package com.hmomen.pneumoscan

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.enableEdgeToEdge
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.hmomen.pneumoscan.ui.theme.PneumoScanTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : ComponentActivity() {

    private lateinit var tflite: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        tflite = Interpreter(loadModelFile("pneumonia_model.tflite"))
        setContent {
            PneumoScanTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(tflite)
                }
            }
        }
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}

@Composable
fun MainScreen(tflite: Interpreter) {
    val context = LocalContext.current
    var selectedImage by remember { mutableStateOf<Bitmap?>(null) }
    var resultText by remember { mutableStateOf("") }
    var confidenceText by remember { mutableStateOf("") }

    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val inputStream = context.contentResolver.openInputStream(it)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            selectedImage = bitmap

            val (prediction, confidence) = runPrediction(bitmap, tflite)
            resultText = "التشخيص: $prediction"
            confidenceText = "نسبة الثقة: ${(confidence * 100).toInt()}%"
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Top
    ) {

        Spacer(modifier = Modifier.height(50.dp))

        Image(
            painter = painterResource(id = R.drawable.peneumoscan_bag), // تأكد من وضع شعار باسم logo.png في res/drawable
            contentDescription = "شعار التطبيق",
            modifier = Modifier.size(120.dp)
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Pneumo Scan",
            style = MaterialTheme.typography.headlineLarge,
            color = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.height(24.dp))


        if (selectedImage != null) {
            Image(
                bitmap = selectedImage!!.asImageBitmap(),
                contentDescription = "الصورة المحددة",
                modifier = Modifier
                    .size(300.dp)
                    .clip(RoundedCornerShape(15.dp))
                    .clickable { imagePicker.launch("image/*") }
            )
        } else {
            Box(
                modifier = Modifier
                    .size(300.dp)
                    .background(MaterialTheme.colorScheme.surface, shape = RoundedCornerShape(15.dp))
                    .clickable { imagePicker.launch("image/*") },
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "اضغط لاختيار صورة",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }
        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = { imagePicker.launch("image/*") },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(text = if (selectedImage == null) "اختر صورة" else "اختر صورة جديدة")
        }
        Spacer(modifier = Modifier.height(24.dp))


        if (resultText.isNotEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(15.dp),
                elevation = CardDefaults.cardElevation(8.dp)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(text = resultText, style = MaterialTheme.typography.headlineMedium)
                    Text(text = confidenceText, style = MaterialTheme.typography.bodyLarge)
                }
            }
        }
    }
}

fun runPrediction(bitmap: Bitmap, tflite: Interpreter): Pair<String, Float> {
    val inputSize = 160

    val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

    val grayscaleBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    for (y in 0 until inputSize) {
        for (x in 0 until inputSize) {
            val pixel = resizedBitmap.getPixel(x, y)
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)
            val gray = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
            val newPixel = Color.rgb(gray, gray, gray)
            grayscaleBitmap.setPixel(x, y, newPixel)
        }
    }

    val input = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(1) } } }
    for (y in 0 until inputSize) {
        for (x in 0 until inputSize) {
            val pixel = grayscaleBitmap.getPixel(x, y)
            val gray = Color.red(pixel) / 255.0f
            input[0][y][x][0] = gray
        }
    }
    val output = Array(1) { FloatArray(1) }
    tflite.run(input, output)
    val predValue = output[0][0]

    return if (predValue <= 0.5) {
        Pair("PNEUMONIA", 1 - predValue)
    } else {
        Pair("NORMAL", predValue)
    }
}