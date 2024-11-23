import pywhatkit
import geocoder

g = geocoder.ip('me')
location = g.latlng  # Get latitude and longitude
message = f"-------------------------------------------------------------Emergency Alert-----------------------------------:Latitude and Longitude: {location}"  # Format as a message

# Send the location as a WhatsApp message
pywhatkit.sendwhatmsg("+919003764344", message, 11, 59, 10)
