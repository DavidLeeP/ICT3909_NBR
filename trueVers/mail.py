import subprocess
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_email(subject, body):
    # Email configuration
    sender_email = ""  # put the email you want to send from here
    receiver_email = "" # put the recipient email here
    password = ""  # get the app password from your mailing account. This is not the normal password for the email account.
    
    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    
    # Add body to email
    message.attach(MIMEText(body, "plain"))
    
    # Create SMTP session
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        
        # Send email
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        print("Email sent successfully")
        
    except Exception as e:
        print(f"Error sending email: {e}")
    
    finally:
        server.quit()

def main():
    # Get current time
    current_time = datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
    
    # Send start notification
    start_subject = "Process Started"
    start_body = f"Process started at {current_time}"
    send_email(start_subject, start_body)
    
    # Path to the script to run - using absolute path
    script_path = "/mnt/megastore/UNI/y3/fyp stuff/completedVers/tuner.py"
    
    # Verify script exists
    if not os.path.exists(script_path):
        error_msg = f"Script not found at path: {script_path}"
        print(error_msg)
        send_email("Script Not Found", error_msg)
        return
    
    try:
        # Use the Python interpreter from the virtual environment
        python_path = os.path.expanduser("~/nbr_env/bin/python")
        
        print(f"Using Python interpreter: {python_path}")
        print(f"Running script: {script_path}")
        
        # Change to the script's directory before running it
        script_dir = os.path.dirname(script_path)
        print(f"Changing to directory: {script_dir}")
        
        # Run the script and capture output
        result = subprocess.run(
            [python_path, script_path],
            capture_output=True,
            text=True,
            check=True,
            cwd=script_dir  # Set the working directory to the script's location
        )
        
        # Get completion time
        completion_time = datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
        
        # Print output to console
        print("\nScript Output:")
        print(result.stdout)
        
        # Prepare completion email
        completion_subject = "Process Completed"
        completion_body = f"{os.path.basename(script_path)} has finished running at {completion_time}"
        
        # Add script output if any
        if result.stdout:
            completion_body += f"\n\nScript output:\n{result.stdout}"
        
        # Send completion notification
        send_email(completion_subject, completion_body)
        
    except subprocess.CalledProcessError as e:
        # Handle script execution errors with more detail
        error_subject = "Process Error"
        error_body = f"Error running {os.path.basename(script_path)}:\n"
        error_body += f"Exit code: {e.returncode}\n"
        error_body += f"Standard output:\n{e.stdout}\n"
        error_body += f"Standard error:\n{e.stderr}"
        
        print("\nError Output:")
        print(error_body)  # Print to console
        send_email(error_subject, error_body)
        print(f"Error running script: {e}")

if __name__ == "__main__":
    main()
