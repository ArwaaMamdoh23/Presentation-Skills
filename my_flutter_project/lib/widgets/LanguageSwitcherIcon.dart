import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';

class LanguageSwitcherIcon extends StatelessWidget {
  const LanguageSwitcherIcon({super.key});

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<Locale>(
      icon: const Icon(
        Icons.language,
        color: Colors.white,
        size: 26,
      ),
      onSelected: (locale) => context.setLocale(locale),
      itemBuilder: (context) => const [
        PopupMenuItem(value: Locale('en'), child: Text('English')),
        PopupMenuItem(value: Locale('ar'), child: Text('العربية')),
        PopupMenuItem(value: Locale('fr'), child: Text('Français')),
        PopupMenuItem(value: Locale('zh'), child: Text('中文')),
        PopupMenuItem(value: Locale('nl'), child: Text('Nederlands')),
      ],
    );
  }
}
