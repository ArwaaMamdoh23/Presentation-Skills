import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import '../widgets/custom_app_bar.dart';
import '../widgets/CustomDrawer.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class FeedbackReportPage extends StatefulWidget {
  const FeedbackReportPage({super.key});

  @override
  State<FeedbackReportPage> createState() => _FeedbackReportPageState();
}

class _FeedbackReportPageState extends State<FeedbackReportPage> {
  Map<String, dynamic>? report;
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    fetchReport();
  }

  Future<void> fetchReport() async {
    final user = Supabase.instance.client.auth.currentUser;
    if (user == null) {
      setState(() {
        isLoading = false;
      });
      return;
    }

    final response = await Supabase.instance.client
        .from('Report')
        .select()
        .eq('User_id', user.id)
        .order('created_at', ascending: false)
        .limit(1)
        .maybeSingle();

    setState(() {
      report = response;
      isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppBar(
        showSignIn: false,
        isUserSignedIn: true,
        backgroundColor: const Color.fromARGB(197, 185, 185, 185),
      ),
      drawer: CustomDrawer(isSignedIn: true),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : report == null
              ? const Center(child: Text('No report found'))
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      sectionTitle('Comprehensive Feedback'),
                      infoRow('Dominant Emotion', report!['dominant_emotion'] ?? ''),
                      infoRow('Dominant Eye Contact', report!['dominant_eye_contact'] ?? ''),
                      infoRow('Emotion Feedback', report!['emotion_feedback'] ?? ''),
                      const SizedBox(height: 20),
                      sectionTitle('Posture Analysis'),
                      infoRow('Dominant Posture', report!['dominant_posture'] ?? ''),
                      infoRow('Meaning', report!['posture_meaning'] ?? ''),
                      infoRow('Posture Feedback', report!['posture_feedback'] ?? ''),
                      const SizedBox(height: 20),
                      sectionTitle('Gesture Analysis'),
                      infoRow('Dominant Gesture 1', report!['dominant_gesture_1'] ?? ''),
                      infoRow('Meaning 1', report!['gesture_meaning_1'] ?? ''),
                      infoRow('Dominant Gesture 2', report!['dominant_gesture_2'] ?? ''),
                      infoRow('Meaning 2', report!['gesture_meaning_2'] ?? ''),
                      infoRow('Gesture Feedback', report!['gesture_feedback'] ?? ''),
                      const SizedBox(height: 20),
                      sectionTitle('Speech Analysis'),
                      infoRow('Detected Language', report!['detected_language'] ?? ''),
                      const SizedBox(height: 20),
                      sectionTitle('Grammar Analysis'),
                      infoRow('Grammar Score', report!['grammar_score'] ?? ''),
                      infoRow('Grammar Feedback', report!['grammar_feedback'] ?? ''),
                      const SizedBox(height: 20),
                      sectionTitle('Speech Pace Analysis'),
                      infoRow('Speech Pace', report!['speech_pace']?.toString() ?? ''),
                      infoRow('Pace Score', report!['pace_score']?.toString() ?? ''),
                      infoRow('Pace Feedback', report!['pace_feedback'] ?? ''),
                      const SizedBox(height: 20),
                      sectionTitle('Fluency Analysis'),
                      infoRow('Fluency Score', report!['fluency_score']?.toString() ?? ''),
                      infoRow('Filler Words', report!['filler_words'] ?? ''),
                      infoRow('Fluency Feedback', report!['fluency_feedback'] ?? ''),
                      const SizedBox(height: 20),
                      sectionTitle('Pronunciation Analysis'),
                      infoRow('Pronunciation Score', report!['pronunciation_score']?.toString() ?? ''),
                      infoRow('Pronunciation Feedback', report!['pronunciation_feedback'] ?? ''),
                      const SizedBox(height: 30),
                      Center(
                        child: Text(
                          'Overall Score: ${report!['overall_score'] ?? ''}',
                          style: const TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                            color: Colors.green,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
    );
  }

  Widget sectionTitle(String title) => Text(
        title.tr(),
        style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.deepPurple),
      );

  Widget infoRow(String label, String value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4.0),
        child: RichText(
          text: TextSpan(
            text: '$label: ',
            style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.black),
            children: [
              TextSpan(
                text: value,
                style: const TextStyle(fontWeight: FontWeight.normal),
              ),
            ],
          ),
        ),
      );
}