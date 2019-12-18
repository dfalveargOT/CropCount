# Bash file to push in github commited changes
git add .
git commit -m "Version 2.0"
echo "Commited Script"
git push origin master -f
echo "Pushed script"
