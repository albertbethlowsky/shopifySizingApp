const { Pool, Client } = require("pg");
//Client - one connection to the database.
//Pool - multiple Client / (connections)

// pools will use environment variables
// for connection information
const pool = new Pool({
  user: "xsapowwe",
  host: "hattie.db.elephantsql.com",
  database: "xsapowwe",
  password: "5Me0XdTbaxt0AL0Rn3Q_z6YGhvhKj2sb",
  port: 5432,
});

pool.query("SELECT NOW()", (err, res) => {
  //console.log(err, res);
  pool.end();
});

const client = new Client({
  user: "xsapowwe",
  host: "hattie.db.elephantsql.com",
  database: "xsapowwe",
  password: "5Me0XdTbaxt0AL0Rn3Q_z6YGhvhKj2sb",
  port: 5432,
});

client.connect();

client.query("SELECT * from users", (err, res) => {
  console.log(res.rows);
  client.end();
});

//create db, connect to db, add data to db, [use data from db]

//link: https://stackoverflow.com/questions/23450534/how-to-call-a-python-function-from-node-js
const spawn = require("child_process").spawn;
//const pythonProcess = spawn('python',["path/to/script.py", arg1, arg2, ...]);

//in python script do:
//print(dataToSendBack)
//sys.stdout.flush()

//in nodejs do this:
//pythonProcess.stdout.on('data', (data) => {
// Do something with the data returned from python script
//});
