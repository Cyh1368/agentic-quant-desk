// PM2 config for the quant-desk research jobs (Lightsail host).
// 4h sentiment cadence aligned to the desk's UTC decision times.
module.exports = {
  apps: [
    {
      name: "qd-sentiment-cycle",
      script: "scripts/run_sentiment_cycle.sh",
      cwd: __dirname,
      interpreter: "bash",
      autorestart: false,           // one-shot per cron tick, not a daemon
      cron_restart: "5 0,4,8,12,16,20 * * *",  // :05 past each decision hour UTC
      out_file: "logs/sentiment-cycle.log",
      error_file: "logs/sentiment-cycle.err",
      time: true,
    },
  ],
};
