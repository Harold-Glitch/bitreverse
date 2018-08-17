#pragma once

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <errno.h>
typedef int socket_t;
#define INVALID_SOCKET -1
#else
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <winsock2.h>
#include <ws2tcpip.h>
//#define close(x) closesocket(x)
#define realpath(N,R) _fullpath((R),(N),_MAX_PATH)
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET socket_t;
#endif

#include <functional>
#include <iostream>
#include <cstdint>
#include <cctype>
#include <sys/stat.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
//#include <unistd.h>
#include <cwctype>
#include <algorithm>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>

#define SERVER_NAME "Unk"
#define SERVER_VERSION "0.0.1"

#define BUFSIZE 8096

using namespace std;

namespace WPP {

	class Request {
	public:
		Request() {

		}
		std::string method;
		std::string path;
		std::string params;
		std::string client;
		std::string hostname;
		map<string, string> headers;
		map<string, string> query;
		map<string, string> cookies;

	private:

	};

	class Response {
	public:
		Response() {
			code = 200;
			phrase = "OK";
			type = "text/html";
			body << "";

			// set current date and time for "Date: " header
			char buffer[100];
			time_t now = time(0);
			struct tm tstruct = *gmtime(&now);
			strftime(buffer, sizeof(buffer), "%a, %d %b %Y %H:%M:%S %Z", &tstruct);
			date = buffer;
		}
		int code;
		string phrase;
		string type;
		string date;
		stringstream body;

		void send(string str) {
			body << str;
		};
		void send(const char* str) {
			body << str;
		};
	private:
	};

	class Exception : public std::exception {
	public:
		Exception() : pMessage("") {}
		Exception(const char* pStr) : pMessage(pStr) {}
		const char* what() const throw () { return pMessage; }
	private:
		const char* pMessage;
		//        const int pCode;
	};

	map<string, string> mime;


	struct Route {
		string path;
		string method;
		void(*callback)(Request*, Response*);
		string params;
	};

	std::vector<Route> ROUTES;

	class Server {
	public:
		void get(string, void(*callback)(Request*, Response*));
		void post(string, void(*callback)(Request*, Response*));
		void all(string, void(*callback)(Request*, Response*));
		void get(string, string);
		void post(string, string);
		void all(string, string);
		bool start(int, string);
		bool start(int);
		bool start();
	private:
		void* main_loop(void*);
		void parse_headers(char*, Request*, Response*);
		bool match_route(Request*, Response*);
		string trim(string);
		void split(string, string, int, vector<string>*);
	};

	void Server::split(string str, string separator, int max, vector<string>* results) {
		int i = 0;
		size_t found = str.find_first_of(separator);

		while (found != string::npos) {
			if (found > 0) {
				results->push_back(str.substr(0, found));
			}
			str = str.substr(found + 1);
			found = str.find_first_of(separator);

			if (max > -1 && ++i == max) break;
		}

		if (str.length() > 0) {
			results->push_back(str);
		}
	}

	string Server::trim(string s) {
		s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(isspace))));
		s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(isspace))).base(), s.end());

		return s;
	}

	void Server::parse_headers(char* headers, Request* req, Response* res) {
		// Parse request headers
		int i = 0;
		char * pch;
		for (pch = strtok(headers, "\n"); pch; pch = strtok(NULL, "\n"))
		{
			if (i++ == 0) {
				vector<string> R;
				string line(pch);
				this->split(line, " ", 3, &R);

				//            cout << R.size() << endl;

				if (R.size() != 3) {
					//                throw error
				}

				req->method = R[0];
				req->path = R[1];

				size_t pos = req->path.find('?');

				// We have GET params here
				if (pos != string::npos) {
					vector<string> Q1;
					this->split(req->path.substr(pos + 1), "&", -1, &Q1);

					for (vector<string>::size_type q = 0; q < Q1.size(); q++) {
						vector<string> Q2;
						this->split(Q1[q], "=", -1, &Q2);

						if (Q2.size() == 2) {
							req->query[Q2[0]] = Q2[1];
						}
					}

					req->path = req->path.substr(0, pos);
				}
			}
			else {
				vector<string> R;
				string line(pch);
				this->split(line, ": ", 2, &R);

				if (R.size() == 2) {
					req->headers[R[0]] = R[1];

					// Yeah, cookies!
					if (R[0] == "Cookie") {
						vector<string> C1;
						this->split(R[1], "; ", -1, &C1);

						for (vector<string>::size_type c = 0; c < C1.size(); c++) {
							vector<string> C2;
							this->split(C1[c], "=", 2, &C2);

							req->cookies[C2[0]] = C2[1];
						}
					}
				}
			}
		}
	}

	void Server::get(string path, void(*callback)(Request*, Response*)) {
		Route r = {
			path,
			"GET",
			callback
		};

		ROUTES.push_back(r);
	}

	void Server::post(string path, void(*callback)(Request*, Response*)) {
		Route r = {
			path,
			"POST",
			callback
		};

		ROUTES.push_back(r);
	}

	void Server::all(string path, void(*callback)(Request*, Response*)) {
		Route r = {
			path,
			"ALL",
			callback
		};

		ROUTES.push_back(r);
	}

	bool Server::match_route(Request* req, Response* res) {
		for (vector<Route>::size_type i = 0; i < ROUTES.size(); i++) {
			if (ROUTES[i].path == req->path && (ROUTES[i].method == req->method || ROUTES[i].method == "ALL")) {
				req->params = ROUTES[i].params;

				ROUTES[i].callback(req, res);

				return true;
			}
		}

		return false;
	}

	void* Server::main_loop(void* arg) {
		int* port = reinterpret_cast<int*>(arg);

		int newsc;

		int sc = (int)socket(AF_INET, SOCK_STREAM, 0);

		if (sc < 0) {
			throw WPP::Exception("ERROR opening socket");
		}

		//        if (!set_nonblocking(sc, 1))
		//            throw WPP::Exception("Can't set socket to non-blocking mode");

		struct sockaddr_in serv_addr, cli_addr;
		serv_addr.sin_family = AF_INET;
		serv_addr.sin_addr.s_addr = INADDR_ANY;
		serv_addr.sin_port = htons(*port);

		if (::bind(sc, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) != 0) {
			//printf("bind failed with error: %d\n", WSAGetLastError());
			throw WPP::Exception("ERROR on binding");
		}

		int optval = 1;
		::setsockopt(sc, SOL_SOCKET, SO_REUSEADDR, (const char *)&optval, sizeof optval);

		listen(sc, 5);

		socklen_t clilen;
		clilen = sizeof(cli_addr);

		while (true) {

			newsc = (int)accept(sc, (struct sockaddr *) &cli_addr, &clilen);

			if (newsc < 0) {
				throw WPP::Exception("ERROR on accept");
			}

			// handle new connection            
			Request req;
			Response res;

			static char headers[BUFSIZE + 1];
			long ret = 0;
			req.client = string(inet_ntoa(cli_addr.sin_addr));

			// And the host name
			struct hostent *hostName;
			struct in_addr ipv4addr;
#ifndef _WIN32
			inet_pton(AF_INET, inet_ntoa(cli_addr.sin_addr), &ipv4addr);
			hostName = gethostbyaddr((const char *)&ipv4addr, sizeof ipv4addr, AF_INET);
			req.hostname = string(hostName->h_name);
			//			printf("Host name: %s\n", hostName->h_name);			

			ret = read(newsc, headers, BUFSIZE);
#else
			inet_pton(AF_INET, inet_ntoa(cli_addr.sin_addr), &ipv4addr);
			hostName = gethostbyaddr((const char *)&ipv4addr, sizeof ipv4addr, AF_INET);
			req.hostname = string(hostName->h_name);

			ret = recv(newsc, headers, BUFSIZE, 0);
#endif              
			if (ret > 0 && ret < BUFSIZE) {
				headers[ret] = 0;
			}
			else {
				headers[0] = 0;
			}

			this->parse_headers(headers, &req, &res);

			if (!this->match_route(&req, &res)) {
				res.code = 404;
				res.phrase = "Not Found";
				res.type = "text/plain";
				res.send("Not found");
			}

			char header_buffer[BUFSIZE];
			string body = res.body.str();
			size_t body_len = strlen(body.c_str());

			// build http response
			sprintf(header_buffer, "HTTP/1.0 %d %s\r\n", res.code, res.phrase.c_str());

			// append headers
			sprintf(&header_buffer[strlen(header_buffer)], "Server: %s %s\r\n", SERVER_NAME, SERVER_VERSION);
			sprintf(&header_buffer[strlen(header_buffer)], "Date: %s\r\n", res.date.c_str());
			sprintf(&header_buffer[strlen(header_buffer)], "Content-Type: %s\r\n", res.type.c_str());
			sprintf(&header_buffer[strlen(header_buffer)], "Content-Length: %d\r\n", (int)body_len);

			// append extra crlf to indicate start of body
			strcat(header_buffer, "\r\n");

			//ssize_t t = 0; 

#ifndef _WIN32
			write(newsc, header_buffer, strlen(header_buffer));
			write(newsc, body.c_str(), body_len);
			close(newsc);
#else
			send(newsc, header_buffer, (int)strlen(header_buffer), 0);
			send(newsc, body.c_str(), (int)body_len, 0);
			closesocket(newsc);
#endif
		}

		return NULL;

	}

	bool Server::start(int port, string host) {
		//         pthread_t worker;

		//         for(int i = 0; i < 1; ++i) {
		//              int rc = pthread_create (&worker, NULL, &mainLoop, NULL);
		//              assert (rc == 0);
		//         }

#ifdef _WIN32
		WSADATA wsd;
		if (WSAStartup(MAKEWORD(2, 2), &wsd) != 0)
			throw WPP::Exception("Can't initialize Winsock2");
#endif

		this->main_loop(&port);

#ifdef _WIN32
		WSACleanup();
#endif

		exit(0);
		return true;
	}

	bool Server::start(int port) {
		return this->start(port, "0.0.0.0");
	}

	bool Server::start() {
		return this->start(80);
	}
}

namespace http
{
	inline int getLastError()
	{
#ifdef _WIN32
		return WSAGetLastError();
#else
		return errno;
#endif
	}

#ifdef _WIN32
	inline bool initWSA()
	{
		WORD sockVersion = MAKEWORD(2, 2);
		WSADATA wsaData;
		int error = WSAStartup(sockVersion, &wsaData);
		if (error != 0)
		{
			std::cerr << "WSAStartup failed, error: " << error << std::endl;
			return false;
		}

		if (wsaData.wVersion != sockVersion)
		{
			std::cerr << "Incorrect Winsock version" << std::endl;
			WSACleanup();
			return false;
		}

		return true;
	}
#endif

	inline std::string urlEncode(const std::string& str)
	{
		static const std::map<char, std::string> entities = {
			{ ' ', "%20" },
			{ '!', "%21" },
			{ '"', "%22" },
			{ '*', "%2A" },
			{ '\'', "%27" },
			{ '(', "%28" },
			{ ')', "%29" },
			{ ';', "%3B" },
			{ ':', "%3A" },
			{ '@', "%40" },
			{ '&', "%26" },
			{ '=', "%3D" },
			{ '+', "%2B" },
			{ '$', "%24" },
			{ ',', "%2C" },
			{ '/', "%2F" },
			{ '?', "%3F" },
			{ '%', "%25" },
			{ '#', "%23" },
			{ '<', "%3C" },
			{ '>', "%3E" },
			{ '[', "%5B" },
			{ '\\', "%5C" },
			{ ']', "%5D" },
			{ '^', "%5E" },
			{ '`', "%60" },
			{ '{', "%7B" },
			{ '|', "%7C" },
			{ '}', "%7D" },
			{ '~', "%7E" }
		};

		static const char hexChars[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };

		std::string result;

		for (auto i = str.begin(); i != str.end(); ++i)
		{
			uint32_t cp = *i & 0xff;

			if (cp <= 0x7f) // length = 1
			{
				auto entity = entities.find(*i);
				if (entity == entities.end())
				{
					result += static_cast<char>(cp);
				}
				else
				{
					result += entity->second;
				}
			}
			else if ((cp >> 5) == 0x6) // length = 2
			{
				result += std::string("%") + hexChars[(*i & 0xf0) >> 4] + hexChars[*i & 0x0f];
				if (++i == str.end()) break;
				result += std::string("%") + hexChars[(*i & 0xf0) >> 4] + hexChars[*i & 0x0f];
			}
			else if ((cp >> 4) == 0xe) // length = 3
			{
				result += std::string("%") + hexChars[(*i & 0xf0) >> 4] + hexChars[*i & 0x0f];
				if (++i == str.end()) break;
				result += std::string("%") + hexChars[(*i & 0xf0) >> 4] + hexChars[*i & 0x0f];
				if (++i == str.end()) break;
				result += std::string("%") + hexChars[(*i & 0xf0) >> 4] + hexChars[*i & 0x0f];
			}
			else if ((cp >> 3) == 0x1e) // length = 4
			{
				result += std::string("%") + hexChars[(*i & 0xf0) >> 4] + hexChars[*i & 0x0f];
				if (++i == str.end()) break;
				result += std::string("%") + hexChars[(*i & 0xf0) >> 4] + hexChars[*i & 0x0f];
				if (++i == str.end()) break;
				result += std::string("%") + hexChars[(*i & 0xf0) >> 4] + hexChars[*i & 0x0f];
				if (++i == str.end()) break;
				result += std::string("%") + hexChars[(*i & 0xf0) >> 4] + hexChars[*i & 0x0f];
			}
		}

		return result;
	}

	struct Response
	{
		bool succeeded = false;
		int code = 0;
		std::vector<std::string> headers;
		std::vector<uint8_t> body;
	};

	class Request
	{
	public:
		Request(const std::string& url)
		{
			size_t protocolEndPosition = url.find("://");

			if (protocolEndPosition != std::string::npos)
			{
				protocol = url.substr(0, protocolEndPosition);
				std::transform(protocol.begin(), protocol.end(), protocol.begin(), ::tolower);

				std::string::size_type pathPosition = url.find('/', protocolEndPosition + 3);

				if (pathPosition == std::string::npos)
				{
					domain = url.substr(protocolEndPosition + 3);
				}
				else
				{
					domain = url.substr(protocolEndPosition + 3, pathPosition - protocolEndPosition - 3);
					path = url.substr(pathPosition);
				}

				std::string::size_type portPosition = domain.find(':');

				if (portPosition != std::string::npos)
				{
					port = domain.substr(portPosition + 1);
					domain.resize(portPosition);
				}
			}
		}

		~Request()
		{
			if (socketFd != INVALID_SOCKET)
			{
#ifdef _WIN32
				int result = closesocket(socketFd);
#else
				int result = close(socketFd);
#endif

				if (result < 0)
				{
					int error = getLastError();
					std::cerr << "Failed to close socket, error: " << error << std::endl;
				}
			}
		}

		Request(const Request& request) = delete;
		Request(Request&& request) = delete;
		Request& operator=(const Request& request) = delete;
		Request& operator=(Request&& request) = delete;

		Response send(const std::string& method,
			const std::map<std::string, std::string>& parameters,
			const std::vector<std::string>& headers = {})
		{
			std::string body;
			bool first = true;

			for (const auto& parameter : parameters)
			{
				if (!first) body += "&";
				first = false;

				body += urlEncode(parameter.first) + "=" + urlEncode(parameter.second);
			}

			return send(method, body, headers);
		}

		Response send(const std::string& method,
			const std::string& body = "",
			const std::vector<std::string>& headers = {})
		{
			Response response;

			if (protocol != "http")
			{
				std::cerr << "Only HTTP protocol is supported" << std::endl;
				return response;
			}

			if (socketFd != INVALID_SOCKET)
			{
#ifdef _WIN32
				int result = closesocket(socketFd);
#else
				int result = ::close(socketFd);
#endif
				socketFd = INVALID_SOCKET;

				if (result < 0)
				{
					int error = getLastError();
					std::cerr << "Failed to close socket, error: " << error << std::endl;
					return response;
				}
			}

			socketFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

#ifdef _WIN32
			if (socketFd == INVALID_SOCKET && WSAGetLastError() == WSANOTINITIALISED)
			{
				if (!initWSA()) return response;

				socketFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
			}
#endif

			if (socketFd == INVALID_SOCKET)
			{
				int error = getLastError();
				std::cerr << "Failed to create socket, error: " << error << std::endl;
				return response;
			}

			addrinfo* info;
			if (getaddrinfo(domain.c_str(), port.empty() ? nullptr : port.c_str(), nullptr, &info) != 0)
			{
				int error = getLastError();
				std::cerr << "Failed to get address info of " << domain << ", error: " << error << std::endl;
				return response;
			}

			sockaddr addr = *info->ai_addr;

			freeaddrinfo(info);

			if (::connect(socketFd, &addr, sizeof(addr)) < 0)
			{
				int error = getLastError();

				std::cerr << "Failed to connect to " << domain << ":" << port << ", error: " << error << std::endl;
				return response;
			}
			/*
			else
			{
			std::cerr << "Connected to " << domain << ":" << port << std::endl;
			}
			*/
			std::string requestData = method + " " + path + " HTTP/1.1\r\n";

			for (const std::string& header : headers)
			{
				requestData += header + "\r\n";
			}

			requestData += "Host: " + domain + "\r\n";
			requestData += "Content-Length: " + std::to_string(body.size()) + "\r\n";

			requestData += "\r\n";
			requestData += body;

#if defined(__APPLE__)
			int flags = 0;
#elif defined(_WIN32)
			int flags = 0;
#else
			int flags = MSG_NOSIGNAL;
#endif

#ifdef _WIN32
			int remaining = static_cast<int>(requestData.size());
			int sent = 0;
			int size;
#else
			ssize_t remaining = static_cast<ssize_t>(requestData.size());
			ssize_t sent = 0;
			ssize_t size;
#endif

			do
			{
				size = ::send(socketFd, requestData.data() + sent, remaining, flags);

				if (size < 0)
				{
					int error = getLastError();
					std::cerr << "Failed to send data to " << domain << ":" << port << ", error: " << error << std::endl;
					return response;
				}

				remaining -= size;
				sent += size;
			} while (remaining > 0);

			uint8_t TEMP_BUFFER[65536];
			const std::vector<uint8_t> clrf = { '\r', '\n' };
			std::vector<uint8_t> responseData;
			bool firstLine = true;
			bool parsedHeaders = false;
			int contentSize = -1;
			bool chunkedResponse = false;
			size_t expectedChunkSize = 0;
			bool removeCLRFAfterChunk = false;

			do
			{
				size = recv(socketFd, reinterpret_cast<char*>(TEMP_BUFFER), sizeof(TEMP_BUFFER), flags);

				if (size < 0)
				{
					int error = getLastError();
					std::cerr << "Failed to read data from " << domain << ":" << port << ", error: " << error << std::endl;
					return response;
				}
				else if (size == 0)
				{
					// disconnected
					break;
				}

				responseData.insert(responseData.end(), std::begin(TEMP_BUFFER), std::begin(TEMP_BUFFER) + size);

				if (!parsedHeaders)
				{
					for (;;)
					{
						std::vector<uint8_t>::iterator i = std::search(responseData.begin(), responseData.end(), clrf.begin(), clrf.end());

						// didn't find a newline
						if (i == responseData.end()) break;

						std::string line(responseData.begin(), i);
						responseData.erase(responseData.begin(), i + 2);

						// empty line indicates the end of the header section
						if (line.empty())
						{
							parsedHeaders = true;
							break;
						}
						else if (firstLine) // first line
						{
							firstLine = false;

							std::string::size_type pos, lastPos = 0, length = line.length();
							std::vector<std::string> parts;

							// tokenize first line
							while (lastPos < length + 1)
							{
								pos = line.find(' ', lastPos);
								if (pos == std::string::npos) pos = length;

								if (pos != lastPos)
								{
									parts.push_back(std::string(line.data() + lastPos,
										static_cast<std::vector<std::string>::size_type>(pos) - lastPos));
								}

								lastPos = pos + 1;
							}

							if (parts.size() >= 2)
							{
								response.code = std::stoi(parts[1]);
							}
						}
						else // headers
						{
							response.headers.push_back(line);

							std::string::size_type pos = line.find(':');

							if (pos != std::string::npos)
							{
								std::string headerName = line.substr(0, pos);
								std::string headerValue = line.substr(pos + 1);

								// ltrim
								headerValue.erase(headerValue.begin(),
									std::find_if(headerValue.begin(), headerValue.end(),
										std::not1(std::ptr_fun<int, int>(std::isspace))));

								// rtrim
								headerValue.erase(std::find_if(headerValue.rbegin(), headerValue.rend(),
									std::not1(std::ptr_fun<int, int>(std::isspace))).base(),
									headerValue.end());

								if (headerName == "Content-Length")
								{
									contentSize = std::stoi(headerValue);
								}
								else if (headerName == "Transfer-Encoding" && headerValue == "chunked")
								{
									chunkedResponse = true;
								}
							}
						}
					}
				}

				if (parsedHeaders)
				{
					if (chunkedResponse)
					{
						bool dataReceived = false;
						for (;;)
						{
							if (expectedChunkSize > 0)
							{
								auto toWrite = min(expectedChunkSize, responseData.size());
								response.body.insert(response.body.end(), responseData.begin(), responseData.begin() + toWrite);
								responseData.erase(responseData.begin(), responseData.begin() + toWrite);
								expectedChunkSize -= toWrite;

								if (expectedChunkSize == 0) removeCLRFAfterChunk = true;
								if (responseData.empty()) break;
							}
							else
							{
								if (removeCLRFAfterChunk)
								{
									if (responseData.size() >= 2)
									{
										removeCLRFAfterChunk = false;
										responseData.erase(responseData.begin(), responseData.begin() + 2);
									}
									else break;
								}

								auto i = std::search(responseData.begin(), responseData.end(), clrf.begin(), clrf.end());

								if (i == responseData.end()) break;

								std::string line(responseData.begin(), i);
								responseData.erase(responseData.begin(), i + 2);

								expectedChunkSize = std::stoul(line, 0, 16);

								if (expectedChunkSize == 0)
								{
									dataReceived = true;
									break;
								}
							}
						}

						if (dataReceived)
						{
							break;
						}
					}
					else
					{
						response.body.insert(response.body.end(), responseData.begin(), responseData.end());
						responseData.clear();

						// got the whole content
						if (contentSize == -1 || response.body.size() >= contentSize)
						{
							break;
						}
					}
				}
			} while (size > 0);

			response.succeeded = true;

			return response;
		}

	private:
		std::string protocol;
		std::string domain;
		std::string port = "80";
		std::string path;
		socket_t socketFd = INVALID_SOCKET;
	};
}
