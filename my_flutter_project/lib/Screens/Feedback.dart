import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';


class FeedbackReportPage extends StatelessWidget {
  const FeedbackReportPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Feedback Report'.tr()),
        backgroundColor: Colors.deepPurple,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            sectionTitle('Comprehensive Feedback'),
            infoRow('Dominant Emotion', 'attentive'),
            infoRow('Dominant Eye Contact', 'Eye Contact'),
            infoRow('Emotion Feedback', 'You appear attentive. Keep your facial expressions engaging.'),

            const SizedBox(height: 20),
            sectionTitle('Posture Analysis'),
            infoRow('Dominant Posture', 'Head Up'),
            infoRow('Meaning', 'Confidence'),
            infoRow('Posture Feedback', 'Great posture! This shows confidence. Keep it up!'),

            const SizedBox(height: 20),
            sectionTitle('Gesture Analysis'),
            infoRow('Dominant Gesture 1', 'Thumbs Up'),
            infoRow('Meaning 1', 'Encouragement'),
            infoRow('Dominant Gesture 2', 'Open Palm'),
            infoRow('Meaning 2', 'Honesty'),
            infoRow('Gesture Feedback', 'A positive gesture. Use it to express approval or encouragement.'),

            const SizedBox(height: 20),
            sectionTitle('Speech Analysis'),
            infoRow('Detected Language', 'en'),

            const SizedBox(height: 20),
            sectionTitle('Grammar Analysis'),
            infoRow('Grammar Score', '6/10 (60%)'),
            infoRow('Grammar Feedback', 'Grammar needs improvement. Review some rules and practice more.'),

            const SizedBox(height: 20),
            sectionTitle('Speech Pace Analysis'),
            infoRow('Speech Pace', '136.5 WPM'),
            infoRow('Pace Score', '100%'),
            infoRow('Pace Feedback', 'Your pace is perfect.'),

            const SizedBox(height: 20),
            sectionTitle('Fluency Analysis'),
            infoRow('Fluency Score', '100/100'),
            infoRow('Filler Words', 'Uh: 0, Um: 0'),
            infoRow('Fluency Feedback', 'Excellent! You used very few filler words. Keep it up!'),

            const SizedBox(height: 20),
            sectionTitle('Pronunciation Analysis'),
            infoRow('Pronunciation Score', '75.5/100'),
            infoRow('Pronunciation Feedback', 'Good pronunciation. A few improvements can make it better.'),

            const SizedBox(height: 30),
            Center(
              child: Text(
                'Overall Score: 7/10',
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
