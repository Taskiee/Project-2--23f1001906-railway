SELECT SUM(units * price) AS total_sales
FROM tickets
WHERE LOWER(TRIM(type)) = 'gold';
