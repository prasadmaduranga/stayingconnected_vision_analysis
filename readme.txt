
# Docker container
docker run -e 'ACCEPT_EULA=Y' -e 'SA_PASSWORD=test@123' \
   -p 1433:1433 --name sqlserver_stayingconected_VA \
   -v sqlserverdata:/var/opt/mssql \
   -d mcr.microsoft.com/mssql/server:2019-latest


# DB insert queries

INSERT INTO [user] (name, age, effected_hand, dominant_hand, gender)
VALUES ('Prasad', 32, 'right', 'right', 'male');

INSERT INTO session (session_type, session_number, user_id)
VALUES ('vision_assessment', 1, 1);

INSERT INTO recording (task, date, time, flipped, session_id)
VALUES ('drinking_water', CAST(GETDATE() AS DATE), CAST(GETDATE() AS TIME), 0, 1);

