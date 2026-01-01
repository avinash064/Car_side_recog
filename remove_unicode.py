# Quick script to remove unicode
with open('extract_mobilenet_features.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove all emojis and special characters
content = content.replace('✅', '[OK]')
content = content.replace('📂', '')
content = content.replace('🔄', '')
content = content.replace('📐', '')
content = content.replace('🔍', '')
content = content.replace('💾', '')
content = content.replace('🧪', '')
content = content.replace('📊', '')
content = content.replace('📦', '')
content = content.replace('🎯', '')
content = content.replace('→', '->')  # Right arrow
content = content.replace('→', '->')  # Another variant

with open('extract_mobilenet_features.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Unicode removed from extract_mobilenet_features.py")
