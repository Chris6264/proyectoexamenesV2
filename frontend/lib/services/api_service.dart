import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // ✅ URL del backend en Render
  static Future<String> getBaseUrl() async {
    return "https://project-s2cg.onrender.com";
  }

  // 📸 Función para enviar las imágenes al backend
  static Future<Map<String, dynamic>?> gradeExam(
    File teacherImg,
    File studentImg,
  ) async {
    final baseUrl = await getBaseUrl();

    // 💡 Despierta el servidor (ping)
    await http.get(Uri.parse('$baseUrl/docs'));

    final uri = Uri.parse('$baseUrl/grade');

    var request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('teacher', teacherImg.path))
      ..files.add(
        await http.MultipartFile.fromPath('student', studentImg.path),
      );

    final response = await request.send();

    if (response.statusCode == 200) {
      final body = await response.stream.bytesToString();
      return jsonDecode(body);
    } else {
      print("Error: ${response.statusCode}");
      return null;
    }
  }
}
