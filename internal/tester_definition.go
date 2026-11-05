package internal

import (
	"time"

	"github.com/codecrafters-io/tester-utils/tester_definition"
)

var testerDefinition = tester_definition.TesterDefinition{
	AntiCheatTestCases: []tester_definition.TestCase{},
	ExecutableFileName: "your_program.sh",
	TestCases: []tester_definition.TestCase{
		{Slug: "zf8", TestFunc: testReadSafetensors},
		{Slug: "pq6", TestFunc: testEmbeddings},
		{Slug: "mo0", TestFunc: testAttention},
		{Slug: "hu4", TestFunc: testTransformerBlock},
		{Slug: "wi3", TestFunc: testLogits},
		{Slug: "hh8", TestFunc: testGreedyDecode},
		{Slug: "df2", TestFunc: testSampling},
		{Slug: "rq5", TestFunc: testKVCache},
		// About 15 seconds to download GPT-2's weights, then up to 300 seconds
		// for the learner's program. This stays under the platform's 10-minute
		// limit, so the learner sees the tester's message rather than a cutoff.
		{Slug: "zr7", TestFunc: testRunGPT2, Timeout: 6 * time.Minute},
	},
}
