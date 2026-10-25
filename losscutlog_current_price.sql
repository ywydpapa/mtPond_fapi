ALTER TABLE losscutLog
    ADD COLUMN currentPrice FLOAT DEFAULT NULL AFTER loscutPrice;
