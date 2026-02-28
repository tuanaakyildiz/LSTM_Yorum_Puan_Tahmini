# 1. Use Node.js to build and run the React app
FROM node:20-alpine

# 2. Set the working directory
WORKDIR /app

# 3. Copy package files and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# 4. Copy the rest of the React source code
COPY . .

# 5. Expose the React development port
EXPOSE 3000

# 6. Start the React app
CMD ["npm", "start"]